# PPO 历史、批处理与固定 KV 优化实测

后续已实现 CPU 多进程环境和学习端加速；当前默认配置、完整轮测量与工期见
[并行环境与学习端优化](ppo_parallel_optimization_zh.md)。本文数据保留为上一阶段对照。

2026-09-11。四暗 PPO 的五项数据路径优化已实现，四人共用代码同时覆盖双明。
在 RTX 4090 上，从同一个只读检查点恢复，完成 2560 个新环境动作以及 Policy、Critic
各 3 个 epoch：**一轮从 56.05 秒降到 19.16 秒，约 2.93 倍加速**。
这是包含优化器更新的完整训练轮；纯采样的提速另列，不能混用。
结果、阶段计时及源文件指纹汇总见[机器可读报告](benchmarks/ppo_pipeline_summary_20260911.json)。

后续正式批量复测：48 局、12288 步的一轮为 172.65 → 46.68 秒，约 3.70 倍加速；
优化版连续四轮平均约 237 步/秒。30 亿步的新工期及测量范围见
[正式批量训练时间](four_dark_optimized_training_eta_zh.md)，下文保留 2560 步短轮诊断。

## 已实现的改动

| 原开销 | 当前实现 | 关键约束 |
|---|---|---|
| 每次缓存查询哈希整段历史 | 每局分配唯一实例 ID，每席位持有数组历史；用实例、席位、窗口起点、有效长度索引，神经缓存绑定冻结权重版本 | 新局更换 ID，加载权重或开始学习时失效缓存 |
| 每条历史反复构造小张量 | 观察产生时写入预分配的 256 行连续整数块；样本引用只读数组窗口，学习时批量复制和 gather | 样本引用的旧块存活到样本释放，不能被新观察覆盖 |
| 编码结果逐行拆分后重新 stack | 学习路径始终保留二维编码块，用一次 cat 和批量索引恢复序列 | 编码结果仍参与当前反向，不能跨参数更新缓存旧特征 |
| 每个缺失 token 单独执行 32 层 | 一次处理同席位已经发生的多个观察，新增 token 按 1/2/4/8 分桶，CUDA 批量按 8 对齐 | 保持因果 mask；不预演尚未执行的未来动作 |
| 每层取页、拼装完整 KV | PPO 每个活跃棋局/席位一个固定槽，Triton 内核直接写入、读取槽内 K/V；对增量网络捕获 CUDA Graph | 终局释放槽，权重变化清空；满窗口滑移仍正确重新预填充 |

动作部分将所有合法动作展开成一批，按状态及起点分组计算条件目标分布，统一计算
所选动作 log-prob、完整合法动作熵与 PPO 损失。指标合并后再回 CPU。
数组只包含该席位有权看到的观察，仍由现有观察编码器产生。

本次保持模型结构及训练参数：棋盘 128 + 动作 128、时序宽度 256、32 层、FFN 1024，
Policy 36,324,226 参数，独立 Critic 36,208,897 参数，Layout 17,292,288 参数。
生产默认仍为 48 局、每轮 12288 个环境动作、学习微批 8 个序列、优化器小批 512 个
决策、Policy/Critic 各 3 个 epoch、BF16 与激活重算。模型参数名称和检查点 v6 格式
保持兼容；已有同架构 PPO 检查点可以继续恢复。

## 4090 对照结果

环境为 WSL Ubuntu 24.04、PyTorch 2.11.0+cu130、RTX 4090 24GB。基线使用改动前
保存的源码副本，优化版使用当前 `src`；报告记录实际导入的源码路径和 SHA256。
两者都读取同一 update 4 检查点，恢复 40 局及其未结束历史和随机状态。
40 局 × 每局 64 个新动作 = 2560 个环境动作，平均输入历史 286.48，最长 751。

### 完整训练一轮

| 指标 | 改动前 | 改动后 |
|---|---:|---:|
| 完整一轮耗时 | 56.05 秒 | **19.16 秒** |
| 完整流程环境动作吞吐 | 45.67 步/秒 | **133.63 步/秒** |
| Policy 实际 optimizer step | 15 | 15 |
| Critic 实际 optimizer step | 15 | 15 |
| 峰值已分配显存 | 17.95 GiB | **11.28 GiB** |
| 峰值保留显存 | 19.67 GiB | 11.40 GiB |

耗时下降约 65.8%，峰值已分配显存下降约 37.2%。计时包含采样、三轮策略和价值
学习、优化器更新、图捕获及该次训练循环的日志和退出工作，不包含计时前的检查点加载。
本轮不跨评测点，因此没有对弈评测、参数保存或布阵参数更新；未生成新的模型文件。
不能用这个短轮估计所有长局、布阵训练或定期评测的摊销。

原始报告：[完整轮基线](benchmarks/ppo_pipeline_update_before_4090_20260911.json)、
[完整轮优化版](benchmarks/ppo_pipeline_update_after_4090_20260911.json)。

该完整轮日志中，采样 4.77 秒、Policy 学习 7.11 秒、Critic 学习 6.84 秒，
内层更新共 18.78 秒。学习已占内层更新的约 74%，因此再只优化采样，完整训练
时间也不会按采样倍数下降。

### 分阶段剖析

以下是另外一次同配置诊断，分开测纯采样和不执行优化器 step 的前后向。

| 指标 | 改动前 | 改动后 |
|---|---:|---:|
| 2560 步纯采样，包含冷缓存准备 | 25.22 秒 | **4.36 秒** |
| 热缓存推进 40 步 | 0.367 秒 | **0.046 秒** |
| 512 个样本的策略前后向 | 1.198 秒 | **0.495 秒** |
| 512 个样本的价值前后向 | 1.107 秒 | **0.552 秒** |
| 热采样 40 步的 Python `__hash__` 调用，含其他对象 | 325,037 | **1,827** |
| 热采样 40 步的 `torch.tensor` 调用 | 1,176 | **7** |
| 学习 512 个样本的 `torch.as_tensor` 调用 | 31,527 | **0** |
| 热采样 40 步的 GPU 内核及复制事件 | 14,333 | **1,666** |
| 策略前后向 128 样本子集的 GPU 内核及复制事件 | 14,946 | **5,162** |

采样约 5.79 倍提速。策略增量计算平均有效批量达到 39.31，最大 40；本段策略推理
捕获 2 个 CUDA Graph、重放 61 次。GPU 事件计数包含实际内核及复制，不表示
Tensor Core 利用率。`__hash__` 仍包含枚举、规则对象等调用；专门测试确认数组
分组及 collation 不调用历史记录的 `__hash__` 或逐记录读取接口。

当前这段纯采样中，环境推进、观察维护与重置共约 2.04 秒，占 4.36 秒的约 47%。
后续瓶颈因此更多落在规则/观察处理及两套 32 层网络的多轮学习上。

原始报告：[分阶段基线](benchmarks/ppo_pipeline_before_verified_4090_20260911.json)、
[分阶段优化版](benchmarks/ppo_pipeline_after_verified_4090_20260911.json)。
首次诊断的未加轨迹指纹结果保留在同目录的 `ppo_pipeline_before/after_4090_20260911.json`，
其阶段耗时为 26.52/5.04 秒；本表使用补充完整调用计数后的复测。

BF16 的计算次序变化会产生舍入差异。复测中的观察/动作轨迹指纹不同，不能称为
逐动作完全相同的轨迹；检查点、随机状态、采样数量及上述历史长度统计相同。
因此时间数据是同起点、同工作量的短轮比较，并非多次统计的置信区间。
数学一致性使用下述固定输入测试单独验证，不用自对弈 loss 相近替代验证。

## 正确性与恢复验证

- WSL CPU 全量回归：240 项通过、5 项跳过、273 项子测试通过。
- CUDA 专项：2 项通过，覆盖直接槽内注意力、图重放及窗口滑移。
- 数组历史与原记录逐项核对，覆盖三种模式、有/无阵亡特征、旧样本生命周期和检查点重放。
- 批量动作概率、熵、PPO 损失及所有参数梯度与原分组实现进行 FP32 数值比较。
- 注入中途 KV 写入失败，验证已提交前缀仍有效、重试正确；验证权重加载和版本变化失效。
- 两进程 CPU/Gloo：实际训练至第一评测点，与旧模型进行 4 局截断对弈，再从保存点
  恢复到第二评测点并再测 4 局；原始棋局历史及模型参数精确恢复，两 rank 最终参数相同。
  这些短局检验调用和存档流程，不提供棋力结论。

正式 32 层模型另做固定输入 CUDA 检查：Policy、Critic 分别以 FP32/BF16 检查
96 次查询，包括多 token 补历史及图重放，对照关闭增量缓存的整段计算。

| 模型/精度 | 最大 context 绝对误差 | 最大输出绝对误差 |
|---|---:|---:|
| Policy / FP32 | 2.27e-6 | 9.54e-7（合法动作 log-prob） |
| Critic / FP32 | 2.27e-6 | 5.24e-10（value） |
| Policy / BF16 | 0.0229 | 0.00513（合法动作 log-prob） |
| Critic / BF16 | 0.0188 | 7.63e-6（value） |

FP32 检查容差 5e-5；BF16 检查容差 0.04。BF16 不保证位级相同，也没有用这些
检查宣称棋力提高。独立报告：[正式模型数值核验](benchmarks/ppo_pipeline_attention_validation_4090_20260911.json)、
[两进程训练与恢复](benchmarks/ppo_pipeline_distributed_validation_20260911.json)。

## 开启方式和实现位置

`configs/bootstrap.yaml` 已默认开启，无需更改训练命令：

```yaml
ppo:
  array_history: true
  fixed_kv: true
  cuda_graphs: true
```

诊断可分别传 `--no-ppo-array-history`、`--no-ppo-fixed-kv`、`--no-ppo-cuda-graphs`。
这些开关只用于局部 A/B；关闭三者并不能撤销批量 collation 等所有优化，完整基线应
使用改动前源码。CUDA 安装了 Triton 时使用编译内核及 CUDA Graph；无 Triton 或
CPU/NPU 使用可移植注意力后备路径，其吞吐不能套用此处 CUDA 结果。
本次编译范围是固定 KV 的 Triton 内核，图捕获范围是增量网络；没有开启整个学习器
的 `torch.compile`。

主要实现：[数组历史](../src/junqi/training/history_arrays.py)、
[固定席位 KV 与图管理](../src/junqi/training/fixed_kv.py)、
[直接访问 KV 的 CUDA 内核](../src/junqi/training/fixed_kv_kernels.py)、
[批量编码与动作头](../src/junqi/training/models.py)、[PPO 采样和损失](../src/junqi/training/ppo.py)。
图捕获的静态输入、预热及张量生命周期遵循
[PyTorch 2.11 CUDA Graph 说明](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)。

可重复验证脚本如下；从仓库根目录在 WSL 运行，使用安装了项目训练依赖的 Python，
`--run-dir` 必须选择新的隔离目录。计时脚本读旧参数但不保存新参数；双进程脚本仅在
小模型评测点保存验证用参数。

```bash
PYTHONPATH=src python tools/benchmark_ppo_pipeline_update.py \
  --checkpoint runs/benchmarks/compact128_g40_exclusive_four_dark_20260911/four_dark/with_dead_rules/checkpoints/latest.pt \
  --run-dir runs/benchmarks/ppo_pipeline_recheck \
  --output runs/benchmarks/ppo_pipeline_recheck.json

PYTHONPATH=src python tools/profile_compact_ppo_bottlenecks.py \
  --checkpoint runs/benchmarks/compact128_g40_exclusive_four_dark_20260911/four_dark/with_dead_rules/checkpoints/latest.pt \
  --moves-per-game 64 --run-dir runs/benchmarks/ppo_pipeline_profile_recheck \
  --output runs/benchmarks/ppo_pipeline_profile_recheck.json

PYTHONPATH=src CUDA_VISIBLE_DEVICES= python -m torch.distributed.run \
  --standalone --nproc-per-node=2 tools/verify_ppo_pipeline_distributed.py \
  --run-dir runs/benchmarks/ppo_pipeline_ddp_recheck \
  --output runs/benchmarks/ppo_pipeline_ddp_recheck.json
```

## 保存策略与 15 天目标

继续采用仅评测时保存：默认每 5000 万个环境动作，与旧最优模型测试 100 局，在完整
训练更新边界的评测前后保存；平时和退出时不增加存盘。中断后从最近评测点恢复，
首次评测前尚未落盘的训练进度不会恢复。本次未启动长期训练，也未改写已有基线检查点。

30 亿是实际环境动作，不是预训练输入 token。15 天连续运行要求至少 2315 环境步/秒，
按 80% 可运行时间要求约 2894 步/秒。本次完整短轮实测为 4090 的 134 步/秒，
**尚不能据此承诺单 PRO 6000 半个月完成**。这五项优化已经消除了大量查询和搬运，
进一步接近目标仍需目标卡实测、规则推进并行和学习器/架构优化。
尚未实施的架构候选见[15 天方案](four_dark_pro6000_15day_plan_zh.md)。
