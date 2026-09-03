# 三模式训练设置、资源需求与运行手册

版本：0.10  
日期：2026-09-03  
适用配置：`configs/bootstrap.yaml` revision 13

## 1. 三个独立训练模式

| 模式 | CLI 名称 | 人数 | 玩家可见棋盘 | 独立默认目录 |
|---|---|---:|---:|---|
| 四暗 | `four_dark` | 4 | 129 点；自己精确，其余未确定时只见相对归属 | `runs/four_dark/<variant>` |
| 双明 | `double_open` | 4 | 129 点；自己和对家盟友精确，两侧敌方未确定时隐藏 | `runs/double_open/<variant>` |
| 二人军棋 | `two_player` | 2 | 60 点；自己精确，对手未确定时编码为 2 | `runs/two_player/<variant>` |

三种模式共享经过同一组单元测试的模型和 trainer 核心，但分别具有训练入口、推理入口、日志、优化器状态、参考模型和检查点。`<variant>` 为 `with_dead_rules` 或 `without_dead_rules`：前者包含确定身份和阵亡先验，后者连 25/50/75 位输入分支都不存在。不得跨模式或跨变体覆盖检查点；加载时会同时校验二者并拒绝错误文件。

每个模式的一个训练进程只实例化一套当前玩家网络：`1 x Policy + 1 x Layout`。四暗/双明的四个座位和二人模式的两个座位只维护各自的主视角历史，不持有各自的网络；布局对所有座位批量生成，策略请求跨环境批量送入同一个 Policy。因此下文参数量和玩家推理显存只计算一次，绝不乘以玩家数。训练另常驻 `1 x reference Policy + 1 x reference Layout` 供 KL 计算，它们不是玩家模型；同步 rollout 使用当前实例的冻结阶段和已保存 old log-prob，不额外复制行为网络。推理进程只加载当前 Policy/Layout 各一份。

## 2. 固定的 Game-GRPO 设置

每个基础对局的每一步都是训练锚点。在每个锚点：

1. 从冻结旧策略的 masked categorical 分布独立有放回采样 `K=4` 个根动作；
2. 每个根动作复制 `M=2` 份环境；
3. 共 `8` 条续局，首步执行对应根动作，后续每一步仍按冻结旧策略概率采样；
4. 每条续局必须运行至规则终局；连续 60 步无子力交互返回和棋 0；
5. 同一根候选的两次终局结果先平均，再在 4 个候选槽中标准化；
6. GRPO 只更新根动作概率，后缀动作不继承根优势；
7. 采样本身已服从旧策略，loss 不再额外乘一次旧策略概率。

| 参数 | 默认值 |
|---|---:|
| 根候选 `K` | 4 |
| 每候选终局副本 `M` | 2 |
| 每锚点终局续局 | 8 |
| 全局 Policy batch | 128 个锚点 |
| 根候选回报/更新 | 512 |
| 终局续局/更新 | 1,024 |
| Policy microbatch，RTX 4090 24GB | bootstrap 16；main 8；extended 6 |
| Actor 推理子批 | 64 |
| 布局全局 batch 目标 | 1,024；当前在线实现每 8 次策略更新消费 64 个已终局布局样本 |
| Policy/Layout 优化器 | AdamW，betas=(0.9,0.95)，weight decay=0.05 |
| Policy 初始学习率 | 1e-4，2,000 update warmup 后余弦下降，KL 超限自动减半 |
| Layout 初始学习率 | 5e-5 |
| GRPO clip | 0.2；主训练稳定后建议 0.15 |
| KL 系数/目标 | 0.02 / 0.015 每动作 |
| 熵系数 | 初始 0.01 |
| 梯度范数裁剪 | 1.0 |
| Policy epoch | 最多 3；KL 或 clip fraction 超限提前停止 |
| 精度 | CUDA BF16；不支持 BF16 时自动改 FP16 并启用动态 GradScaler；归约使用 FP32 |
| 历史 | 固定初始棋盘 token + 最近最多 1,000 个动作后转移 token |
| 检查点 | 默认 `latest.pt` 每 10 update 原子更新；每 500 update 留档，默认保留 10 份；当前二人长跑覆盖为每 1/100 update，保留 4 份归档 |

## 3. 可选模型规模

参数量为当前 PyTorch 实现实际统计值，而不是文档估算；表中每个模型只计一份，不因二人或四人座位数乘 2/4。

| `--model-scale` | 带死规则 Policy | 不带死规则 Policy | Layout | 带死规则合计 | 用途 |
|---|---:|---:|---:|---:|---|
| `bootstrap` | 26,610,696 | 26,459,656 | 8,887,296 | 35,497,992 | 正确性、吞吐和早期学习验证 |
| `main` | 144,157,704 | 144,006,664 | 17,292,288 | 161,449,992 | 正式强棋力训练 |
| `extended` | 215,534,600 | 215,383,560 | 25,697,280 | 241,231,880 | 仅在主模型明确欠拟合时使用 |

Policy 包含 139 类棋盘码 embedding、256 维带道路/铁路图关系偏置的 BoardEncoder、256 维公开动作编码、动作与棋盘状态 concat 得到的 512 维 token、因果历史 Transformer，以及“起点 -> 条件终点”两阶段动作头。带死规则版本额外包含规范 75 位确定阵亡先验投影和棋盘/阵亡 concat 融合，共增加 151,040 个参数；关闭版本没有这两个模块。Layout 是固定 25 枚棋子序列驱动的 256 维自回归 Pointer Transformer；这同一序列也定义开启态每名对手的 25 位阵亡槽。

下文 RTX 4090 容量探针采自扩展棋盘码表与阵亡融合之前的同构模型；带死规则 main Policy 相对当时的 143,992,584 增加 165,120 个参数，仅约 0.115%，BF16 权重增量约 323 KiB，因此原实测值仍可作为两种变体的起测档。正式长跑仍应按当前 revision 13 代码重新做 1,001-token OOM 上探。

## 4. 每种模式的模拟量

仓库随机规则基准中，二人平均局长约 327.3 步、续局平均剩余 182.5 步；四国平均局长约 610.6 步、续局平均剩余 334.5 步。双明暂以四暗的同等局长作保守容量规划；实际通常可能更短，但必须由训练中的滚动数据修正。

| 模式 | 终局续局/基础局 | 分叉环境步/基础局 | 分叉环境步/默认 update |
|---|---:|---:|---:|
| 二人 | 约 2,618 | 约 0.430M | 约 0.187M |
| 四暗 | 约 4,885 | 约 1.494M | 约 0.342M |
| 双明 | 暂按 4,885 | 暂按 1.494M | 暂按 0.342M |

相较已删除的“穷举全部合法根动作、每种模拟四次”方案，二人的终局续局和分叉步分别减少约 15.1/16.8 倍，四国分别减少约 21.2/23.2 倍。

### 4.1 当前 RTX 4090 二人端到端状态

2026-09-03 检查结果：机器只有一张 NVIDIA GeForce RTX 4090 24GB；Windows 驱动 `595.79`、WSL 可见 CUDA `13.2`，`siguozero` 环境使用 PyTorch `2.11.0+cu130` 并可正常访问 GPU。

随后使用当前真实 `two_player + bootstrap + K=4 + M=2` 路径进行了一次端到端探针。所有玩家共享同一 Policy/Layout，Actor 子批为 64；结果如下：

| 锚点 | 终局续局 | 分叉环境步 | rollout 时间 | 端到端 update 时间 | 吞吐 |
|---:|---:|---:|---:|---:|---:|
| 8 | 64 | 22,011 | 158.51 s | 160.38 s | 138.86 步/s |

这是一次从初始基础局出发的小样本实测，不是长期稳定吞吐承诺。它已经包含真实规则推进、逐步概率采样和完整终局，但只有 bootstrap 模型且样本较小；长历史分布、main 模型、热稳定性、评测和故障恢复只能使当前代码的规划时间更保守。探针期间 GPU 利用率快照约 9%，说明主要瓶颈在 Python 环境、局面组织和每步重复编码，而不是 Tensor Core 或显存已满。

同日已按用户指令启动正式二人主模型进程：`two_player + main + dead_rules + microbatch=24 + actor_batch=64`，累计停止条件为 `3,000,000,000` 个分叉续局步。所有座位共享同一套 Policy/Layout。启动后首先原子保存 update 0；此后每个完成的 update 覆盖 `latest.pt`，每 100 update 归档一次并只保留最近 4 份。独立心跳每 15 秒采集 GPU 利用率、显存、温度、功耗、进程显存及磁盘余量。启动初期处于终局 rollout 阶段时，实测快照约为 `48%～58% GPU、2.9～3.0 GiB、54～60°C、144～226 W`；这不是 learner 反向阶段的显存峰值。

`microbatch=24` 是 requested 起点而不是保证值。最长 1,001-token 二人 learner 容量探针曾测得约 `19.46 GiB`，因此理论上可以运行；真实长跑若因碎片或峰值 OOM，trainer 会在尚未执行 optimizer step 的整轮上自动清梯度并按 `24 -> 12 -> 6 -> 3 -> 1` 回退，同时把实际值写入日志和检查点，绝不跳过数据或从头重训。

## 5. 分模式累计训练预算

三个阶段都保留相同模型输入和终局规则，只改变累计样本量与评测强度。

| 模式/阶段 | learner updates | 锚点 | 终局续局 | 分叉环境步 |
|---|---:|---:|---:|---:|
| 二人冷启动 | 约 7.8K | 1M | 8M | 约 1.46B |
| **二人当前优先里程碑** | **约 16.1K** | **约 2.055M** | **约 16.44M** | **3.00B** |
| 二人主训练 | 约 62.5K | 8M | 64M | 约 11.68B |
| 二人顶尖容量 | 200K-600K | 25.6M-76.8M | 0.205B-0.614B | 约 37.4B-112.1B |
| 四暗冷启动 | 约 7.8K | 1M | 8M | 约 2.68B |
| 四暗主训练 | 约 62.5K | 8M | 64M | 约 21.41B |
| 四暗顶尖容量 | 200K-600K | 25.6M-76.8M | 0.205B-0.614B | 约 68.5B-205.5B |
| 双明冷启动 | 约 7.8K | 1M | 8M | 暂按 2.68B |
| 双明主训练 | 约 62.5K | 8M | 64M | 暂按 21.41B |
| 双明顶尖容量 | 200K-600K | 25.6M-76.8M | 0.205B-0.614B | 暂按 68.5B-205.5B |

“顶尖容量”是预留上限，不是棋力保证。每 10K-25K updates 应在训练外进行历史检查点回归、固定基线、座位轮换、和棋率、策略熵和人类盲测。在线训练局仍由同一 current 模型控制全部座位。曲线饱和或退化时应停下来诊断，而不是机械烧完样本。

当前 30 亿步目标只适用于首先运行的二人模式，并严格按 `cumulative/continuation_plies` 计数。默认配置每个外层 update 约产生 `1024 x 182.5 = 186,880` 个二人分叉步，因此预计需要 `ceil(3e9 / 186880) = 16,054` 个完整 update；实际局长会变化，最终必须按累计步数停止，而不能只依赖这个 update 近似值。最多 3 个 Policy epoch 意味着这批数据约产生 `16,054～48,162` 次 Policy optimizer step，但不会增加 rollout 步数。

## 6. GPU 显存建议

训练使用 FP32 master 参数/AdamW 状态和 CUDA autocast；激活 checkpoint 默认开启。分叉续局数量下降减少累计时间，但不会等比例降低单个 1,001-token 样本的峰值显存。新增阵亡融合的参数、梯度及 AdamW 状态约为 2.3 MiB，另加每个有效历史 token 75 个二值输入（当前 collator 用 FP32，约 300 bytes），不改变下表建议档位。

| GPU | bootstrap Policy microbatch | main | extended | 建议用途 |
|---|---:|---:|---:|---|
| RTX 4090 24GB | 16（实测 24 可运行） | 8（实测 12 可运行） | 6（实测 8 可运行） | 开发、冷启动、单机实验 |
| 48GB | 32-48 | 20-24 | 12-16 | 中等规模 actor/learner；必须复测 |
| RTX PRO 6000 Blackwell 96GB | 64-96 | 48-64；二人可从 64 起测、再测 96 | 32-40 | 96GB GDDR7 ECC；必须按最长上下文复测 |
| H100 80GB | 64 起 | 32 起 | 24 起 | 正式训练起点；必须复测 |
| H200 141GB | 128 起 | 64 起 | 48 起 | 更大的 Actor 驻留和 learner microbatch；必须复测 |

当前机器实测环境是 WSL2 Ubuntu 24.04、RTX 4090 24GB、PyTorch 2.11.0+cu130；带死规则的三种模式均已执行 CUDA 冒烟测试，不带死规则的二人模式也已完成 CUDA 训练、断点恢复和推理闭环。下表是 revision 13 前一同构版本的四暗、1,001 token、BF16、activation checkpoint 容量探针；当前 Policy/Layout 各一份与 KL reference Policy/Layout 各一份常驻，Layout AdamW 状态已分配，不存在四份座位模型：

| 模型 | microbatch | 单步时间 | CUDA 峰值分配 |
|---|---:|---:|---:|
| bootstrap | 1 / 8 / 16 / 24 | 0.59 / 2.57 / 4.98 / 7.06 s | 1.71 / 7.09 / 13.79 / 20.48 GiB |
| main | 1 / 4 / 8 / 12 | 0.89 / 2.40 / 4.39 / 6.30 s | 3.55 / 7.18 / 12.68 / 18.20 GiB |
| extended | 1 / 8 | 1.22 / 6.49 s | 5.32 / 17.72 GiB |

默认值故意低于实测极限，为 Actor 批处理、分配碎片、日志和局面长度差异留出余量。48/80/141GB 行是按实测斜率给出的起测值，并非在这些卡上的实测结论；最终 microbatch 必须用目标机器和真实长度分桶复测，至少保留 10% 显存余量。

仓库提供可重复的真实前向/反向/AdamW 容量探针；它会保留一份当前 Policy/Layout 及一份 KL reference Policy/Layout，展开指定长度的合法玩家视角状态，并报告单步时间和 CUDA 峰值：

```bash
python tools/benchmark_cuda.py --mode four_dark \
  --model-scale bootstrap --context-tokens 1001 --batch-size 1
```

在命令末尾添加 `--no-dead-rules` 可测关闭态；不写或写 `--dead-rules` 测开启态。

建议依次测试 `128/256/512/1001` token，再递增 `--batch-size`；不要用短开局的显存结果直接决定最长上下文 batch。

## 7. H100 等效时间

估算按每张 H100 对小矩阵、图编码和环境交错负载持续约 100 TFLOPs，并对 learner、Python/原生环境差异、通信、负载不均、评测和故障恢复加入 2-3 倍规划余量。

| 模式/阶段 | 预计 H100-GPU-days |
|---|---:|
| 二人冷启动 | 约 2-3 |
| 二人主训练 | 约 9-14 |
| 二人顶尖容量 | 约 29-128 |
| 四暗冷启动 | 约 9-13 |
| 四暗主训练 | 约 58-87 |
| 四暗顶尖容量 | 约 185-834 |
| 双明 | 暂按四暗同档规划，训练后用真实局长修正 |

单独训练一个四国模式达到顶尖容量时，8 张 H100 约为 23-105 天，16 张约为 12-53 天。二人模式相同预算约为 4-16 天（8 张）或 2-8 天（16 张）。若三个模式都训练独立顶尖模型，总量约为 400-1,800 H100-GPU-days；这仍不意味着必须采购 64 张卡，8-16 张通过排队或并行分配即可，只是墙钟时间不同。

RTX 4090 与 H100 在 BF16、显存、互联和小批利用率上差异很大，不能只按理论峰值换算。单张 4090 适合先完成三模式冷启动和性能剖析；正式四国顶尖训练建议至少 4-8 张 80GB 级 GPU，想压到数周再考虑 16 张。

### 7.1 二人/双明 RTX 4090 最长上下文实测

main 模型、1,001 token、BF16、activation checkpoint、共享 current 与 KL reference 模型对常驻时，本机 learner 单步实测为：

| 模式 | microbatch | 前向+反向+AdamW | CUDA 峰值分配 |
|---|---:|---:|---:|
| 二人 | 8 | 3.54 s | 7.59 GiB |
| 二人 | 16 | 6.55 s | 13.53 GiB |
| 二人 | 24 | 9.23 s | 19.46 GiB |
| 双明 | 8 | 5.27 s | 12.68 GiB |
| 双明 | 12 | 7.46 s | 18.20 GiB |

这些是 learner 容量探针，不含终局规则 Actor。H100 80GB、全局 128 锚点的保守起测 microbatch 仍取 bootstrap/main/extended=`64/32/24`，对应约 `2/4/6` 次梯度累积；二人模式可在稳定后向上探测 `96/64/32`。H100 数字尚未在目标卡实测，必须保留 10% 显存并逐级 OOM 探测。

### 7.2 当前二人 30 亿步的实际时间判断

30 亿步作为二人“冷启动后扩大验证”里程碑是合理的：它约为冷启动分叉步预算的 `2.05` 倍，但只有正式主训练预算的 `25.7%`，所以不能被描述为顶尖棋力训练终点。rev.14 已启用增量 KV、持久 COW 历史、packed 棋盘、长度分桶和规则静态查表；时间取决于端到端 Actor 吞吐，而不是 learner microbatch：

| 二人端到端吞吐 | 30 亿步纯运行时间 | 按 90% 可用率 | 结论 |
|---:|---:|---:|---|
| rev.14 main 自然终局实测 431.37 步/s | 80.5 天 | 89.4 天 | 加 learner/评测后先按 100～115 天 |
| 1,000 步/s | 34.7 天 | 38.6 天 | 约 6 周，仍需实测原生 Actor/更快 GPU |
| 工程启动门槛 2,000 步/s | 17.4 天 | 19.3 天 | 达标后可以启动阶段训练 |
| 优化目标 3,000 步/s | 11.6 天 | 12.9 天 | 可在约两周内完成纯采样 |

实测命令使用 `tools/benchmark_rollout.py`，完整训练栈常驻，`64` 条续局共 `22,327` 步、`51.76 s`，峰值 CUDA 分配 `9.22 GiB`；固定 256 步的同种子 A/B 从 `179.6` 提升到 `435.4` 步/s（`2.42x`）。首个完整 update 结束后，使用 `metrics.jsonl` 中的真实 `rollout/plies_per_second` 重新估算，之后用至少 10 个 update 的移动中位数更新 ETA：

$$
T_{days}=\frac{3\times10^9-N_{done}}{v_{plies/s}\times86400\times availability}.
$$

即使达到 2,000 步/s，正式排期也应按约 `3～4` 周墙钟预留。当前代码已经支持单模式 `torchrun/DDP`，并实现增量 KV 与分支前缀 copy-on-write；第二张或第四张 GPU 会取得独立环境分片并同步 learner 梯度。继续达到该吞吐仍需要自定义 paged/CUDA Graph actor 或原生批量规则环境，并评估专用 actor GPU 上的一版本异步流水，不能把 DDP 卡数直接视为线性加速倍数。

### 7.3 RTX PRO 6000 Blackwell 96GB 时间规划

NVIDIA 官方规格中，[RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) 为 `24 GB GDDR6X、83 FP32 TFLOPS、1,321 AI TOPS`；[RTX PRO 6000 Blackwell Workstation Edition](https://www.nvidia.com/en-au/products/workstations/professional-desktop-gpus/rtx-pro-6000/) 为 `96 GB GDDR7 ECC、1,792 GB/s、125 FP32 TFLOPS、4,000 AI TOPS、600 W`。其中 AI TOPS 采用的精度/稀疏口径不能直接等同本项目 BF16 速度；较可比的 FP32 峰值只约 `1.51x`，显存带宽约为 RTX 4090 的 `1.78x`，而容量为 `4x`。

按二人最长上下文的实测显存斜率，96GB 卡建议 Policy `microbatch=64` 起步，稳定后测 `96`；全局 batch 仍为 128，所以从 4090 上每 epoch 约 6 个物理子批降至 2 个。Actor inference batch 可从 `256` 起测，视 GPU 利用率和峰值显存测到 `512`。增大这些批次主要降低调度/learner 开销，完全不改变 30 亿环境步、约 16.44M 条终局续局和约 16,054 个外层 update。

增量 KV 完成后，自然终局实测中约 `86%` 墙钟仍在 Policy inference、约 `12%` 是 Python 环境工作；因此 96GB 可以把 actor batch 从 `64` 向 `128/256` 探测，但不能按容量 `4x` 直接换算。以当前 workload 的瓶颈结构估算：

| RTX PRO 6000 方案 | 目标端到端吞吐（未实测） | 30 亿步墙钟规划（含 learner/评测） |
|---|---:|---:|
| 1× RTX PRO 6000 96GB | `750～1,050` 步/s | 约 `45～60` 天 |
| 2× RTX PRO 6000 96GB，DDP 效率 85%～90% | `1,300～1,850` 步/s | 约 `26～38` 天 |

两行都是根据 4090 实测瓶颈作的工程区间，不是目标卡 benchmark。96GB 最大价值是能把更多环境/KV 与较大批量常驻显存，为后续优化创造条件，而不是自动把 Python 串行环境加速四倍。正式 microbatch 建议 main 从 `64` 开始，actor batch 从 `128` 开始，各自逐级测到 `96/256`；全局锚点 batch 仍为 128，不能让每卡都重复 128 个锚点。

### 7.4 RTX 4090 的 1/2/4 卡优化后时间规划

先按 rev.14 单卡实测 `431.37` 步/s、2 卡端到端效率 `88%`、4 卡效率 `75%` 外推
30 亿 continuation plies。learner、评测、检查点和 90% 可用率已作为区间余量加入：

| 配置 | 预计聚合吞吐 | 30 亿步纯 rollout | 工程墙钟规划 |
|---|---:|---:|---:|
| 1× RTX 4090 | 431 步/s（实测） | 80.5 天 | **100～115 天** |
| 2× RTX 4090 | 约 759 步/s | 45.7 天 | **55～65 天** |
| 4× RTX 4090 | 约 1,294 步/s | 26.8 天 | **32～40 天** |

多卡吞吐是推算，必须在目标机器用同一 `benchmark_rollout.py` 和一次完整 update
复测。四卡时 24 个 CPU 逻辑核也可能成为约束。

同一 rev.14 负载的高端卡单卡规划如下。NVIDIA 官方规格为：H100 SXM `80GB /
3.35TB/s`，B200 `180GB / up to 8TB/s`，B300 `288GB / up to 8TB/s`；B300 的
attention layer acceleration 官方称相对 Blackwell 最高 `2x`。这些优势不能完整
映射到本项目，因为当前约 12% 时间已在 Python 裁判，而且大量 SDPA 是小 batch：

| 每 rank 一张卡 | 建议 main learner microbatch 起点 | actor batch 起点 | 推算吞吐 | 30 亿步工程墙钟 |
|---|---:|---:|---:|---:|
| H100 SXM 80GB | 32 | 128 | 850～1,200 步/s | 36～55 天 |
| B200 SXM 180GB | 64 | 256 | 1,100～1,500 步/s | 29～43 天 |
| B300 SXM 288GB | 96 | 256 | 1,200～1,650 步/s | 27～39 天 |

来源：[NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/)、
[NVIDIA HGX H100/H200/B200 规格](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)、
[NVIDIA B300 企业参考架构](https://docs.nvidia.com/enterprise-reference-architectures/whitepaper/hgx-servers-and-spectrum-x.pdf)、
[NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-for-the-era-of-ai-reasoning/)。
这里的时间全部是项目推算，不是 NVIDIA 发布的 SiguoZero benchmark。

以既有 H100-GPU-day 预算为基准，暂取一张 4090 对本混合负载等效 `0.30` 张 H100，2 卡效率 `90%`、4 卡效率 `80%`：

$$
T_{4090}(n)=\frac{D_{H100}}{0.30\,n\,\eta_n},\qquad
\eta_1=1,\ \eta_2=0.90,\ \eta_4=0.80.
$$

| 模式/累计阶段 | 1×4090 | 2×4090 | 4×4090 |
|---|---:|---:|---:|
| 二人冷启动 | 7-10 天 | 4-6 天 | 2-3 天 |
| 二人主训练 | 30-47 天 | 17-26 天 | 9-15 天 |
| 二人顶尖容量 | 97-427 天 | 54-237 天 | 30-133 天 |
| 双明冷启动 | 30-43 天 | 17-24 天 | 9-14 天 |
| 双明主训练 | 193-290 天 | 107-161 天 | 60-91 天 |
| 双明顶尖容量 | 617-2,780 天 | 343-1,544 天 | 193-869 天 |

各阶段是累计目标，不能相加。当前仓库已实现单模式 DDP learner 和每 rank 独立 Actor 分片，所以 2/4 卡可以共同训练同一个模型；表中仍是完成增量 KV cache、分支前缀共享和原生并行 Actor 后的规划值，不是仅启用 DDP 就能保证的时间。若 4090/H100 实测等效比为 `0.25/0.35`，表中时间分别乘 `1.20/0.86`。

### 7.5 rollout 总量

一个 rollout 指“固定锚点和一个根动作后，一个独立副本继续走到终局”。每个锚点 `K=4,M=2`，全局 batch 为 128，因此：

$$
N_{rollout}=U\times128\times4\times2=1024U.
$$

| 累计阶段 | 每模式 rollout | 二人+双明合计 |
|---|---:|---:|
| 冷启动 | 8M | 16M |
| 主训练 | 64M | 128M |
| 顶尖容量 | 204.8M-614.4M | 409.6M-1.2288B |

30 亿步目标的 rollout 数取决于平均终局剩余长度，而不是固定常数。历史估算
`182.5` 步/rollout 对应约 `16.44M` rollouts 和 `16,054` updates；rev.14 自然终局
探针 `348.9` 步/rollout 对应约 `8.60M` rollouts 和 `8,397` updates。正式先按
`8.6M～16.4M` rollouts 规划，以完成首批 updates 后的实际均值修正。

基础对局和训练外评测局不计入上述 rollout。顶尖容量已经包含冷启动与主训练，不是三阶段相加；模型共享也意味着 rollout 数不再乘以二人/四人的座位数量。

## 8. WSL/conda 安装与运行

仓库提供 `environment.yml`。标准创建方式：

```bash
cd /mnt/e/agent/siguozero
conda env create -f environment.yml
conda activate siguozero
python -m pip install --no-deps --no-build-isolation -e .
```

当前机器已经创建 `/root/anaconda3/envs/siguozero`。可直接运行：

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate siguozero
cd /mnt/e/agent/siguozero
```

三种正式训练入口：

```bash
python -m junqi.training.train_four_dark --device cuda --model-scale bootstrap --dead-rules
python -m junqi.training.train_double_open --device cuda --model-scale bootstrap --dead-rules
python -m junqi.training.train_two_player --device cuda --model-scale bootstrap --dead-rules
```

扩展到主模型时使用 `--model-scale main`。当前 24GB 实测默认是 bootstrap/main/extended 分别 `16/8/6`；H100 可从 `64/32/24` 起测，H200 从 `128/64/48` 起测。`--microbatch` 可覆盖自动档位，`--anchor-batch` 改变逻辑 Policy batch，`--actor-batch` 只改变 CUDA 推理子批，不改变 K=4、M=2 的算法。

单模式多 GPU 使用 `torchrun --standalone --nproc-per-node=N`。`anchor_batch` 是全局值且必须被 `N` 整除，`microbatch` 是每张卡的值；每个 rank 仍只有一套供所有座位共享的 current Policy/Layout。完整约定见[多 GPU 训练与历史编码加速设计](multi_gpu_and_performance_zh.md)。

当前二人 30 亿步任务的等价启动命令为：

```bash
python -m junqi.training.train_two_player \
  --config configs/bootstrap.yaml --device cuda --model-scale main \
  --microbatch 24 --actor-batch 64 --target-continuation-plies 3000000000 \
  --checkpoint-every 1 --archive-every 100 --keep-checkpoint-archives 4 \
  --resource-monitor-seconds 15 --dead-rules
```

## 9. 自动恢复、日志和检查点

每个运行目录包含：

```text
runs/<mode>/
  with_dead_rules/
    resolved_config.json
    train.log
    metrics.jsonl
    latest_metrics.json
    resource_metrics.jsonl
    resource_latest.json
    tensorboard/
    checkpoints/
      latest.pt
      manifest.json
      update_000000500.pt
      ...
  without_dead_rules/
    ...（同样结构，完全独立）
```

新运行在进入第一个昂贵 rollout 前就写入 update 0 检查点。此后 `latest.pt` 使用临时文件加原子替换，保存变体标记、共享的当前 Policy/Layout、两个优化器、阶段参考模型、update、累计锚点/续局/环境步、学习率缩放、待消费布局终局样本、所有 rank 尚未结束的基础对局及历史窗口，以及各 rank 的 Python/基础局/CPU/CUDA RNG。开启态还保存每位玩家的持久确定身份表和阵亡库存；关闭态没有阵亡历史张量。检查点不会为不同座位重复保存权重。单卡收到 SIGINT/SIGTERM 或 Python 异常时可紧急保存；多卡异常为避免不对称 collective 写出损坏状态，保留最后一个完整原子检查点并由 torchrun 终止各 rank。

训练命令默认自动寻找对应模式、对应变体目录中的 `latest.pt` 并继续；不需要额外 `--resume` 参数。`--run-dir` 是基目录，程序自动追加变体名。`--no-resume` 只允许配合全新或空的运行目录使用；非空目录没有 `latest.pt` 时也会拒绝随机重启，避免把损坏的运行误当成新实验。确需新实验时应指定另一个 `--run-directory`。硬断电/SIGKILL 最多损失默认 10 个 updates；当前二人任务覆盖为每 1 update 保存，因此最多重算正在执行的一个原子 update，不会退回随机初始化。恢复后中盘基础局也会从原状态继续。

日志逐 update 记录总 loss、GRPO loss、精确策略 KL、熵、importance ratio、clip fraction、非零优势比例、梯度范数、学习率、胜/和/负续局数、锚点数、终局续局数、分叉/基础环境步、吞吐、耗时，以及 CUDA 当前/峰值显存。独立资源线程不等待 update，在 `resource_metrics.jsonl` 中追加心跳并原子覆盖 `resource_latest.json`，包含当前阶段、PID、有效 microbatch/actor batch、GPU 利用率/显存/温度/功耗、进程 CUDA 分配和磁盘余量；达到 95% 显存、85°C 或磁盘少于 10 GiB 时写警告。TensorBoard 未安装时 JSONL 和文本日志仍正常工作。

## 10. 推理

推理同样按概率采样而不是 argmax，并继续施加合法动作 Hard Mask：

```bash
python -m junqi.training.infer_four_dark \
  --checkpoint runs/four_dark/with_dead_rules/checkpoints/latest.pt --device cuda --games 1

python -m junqi.training.infer_double_open \
  --checkpoint runs/double_open/with_dead_rules/checkpoints/latest.pt --device cuda --games 1

python -m junqi.training.infer_two_player \
  --checkpoint runs/two_player/with_dead_rules/checkpoints/latest.pt --device cuda --games 1
```

Python 集成使用 `InferenceEngine.from_checkpoint()`、`sample_layouts()`、`select_action()` 和 `step()`。一个 `InferenceEngine` 只加载一个 Policy 和一个 Layout，所有座位重复调用它们；模型检查点与请求模式或显式 `--dead-rules/--no-dead-rules` 要求不一致时直接报错。

## 11. 当前实现边界

当前版本是可运行、可反向传播、可中断恢复的 PyTorch 基线，已把候选动作和并行续局推理按 CUDA 子批执行，支持单机多 GPU DDP，并启用 autocast、TF32、fused SDPA 可用路径和 activation checkpoint。rev.14 还实现了增量 causal KV、持久 COW 历史、packed 棋盘、learner 长度分桶及规则静态查表。要达到 2,000+ 步/s 的目标，还必须继续完成：

1. 将 Python 规则 actor 替换为 C++/Rust 或编译向量化环境，并与 Python 裁判持续差分；
2. 以自定义 paged-attention/CUDA Graph 消除增量路径中剩余的 K/V stack/cat；
3. 在现有单机 DDP 上增加带一版本 behavior snapshot 的异步 actor-learner 服务及需要时的多节点/FSDP；
5. 增加训练外历史回归评测调度器、人类评测接口和长期断电演练。

30 亿步任务可以作为阶段训练启动，但不应把一次探针视为交付日期或把 30 亿步视为顶尖棋力保证；以首个完整 update 及其后 10 个 update 的移动中位数持续校准。
