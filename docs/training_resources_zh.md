# 三模式训练设置、资源需求与运行手册

四国 main 的最新 CPU 多进程环境、32 微批和 4090 完整训练工期见
[并行环境与学习端优化](ppo_parallel_optimization_zh.md)。下列旧结构容量表保留原测量条件。

> revision 20 已改为[整盘向量单层投影](whole_board_linear_zh.md)。本文的逐点图编码器、旧参数量和已有性能数字属于修改前版本，不能作为新架构的计时或显存结论。

> 2026-09-10：四暗、双明已改为 PPO + 独立价值模型，当前设置和 RTX 4090
> 实测请读[四国 PPO 运行手册](four_player_ppo_zh.md)。下文 v0.10 的三模式
> Game-GRPO 资源表作为历史基线保留，二人模式的算法仍为 GRPO。

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
4. 每条续局必须运行至规则终局；连续 70 步没有吃子产生和棋，训练奖励为 -0.15，默认无总步数上限；
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
| 检查点 | 四国仅在每 5000 万环境步评测时保存；二人默认每 10 update 更新、500 update 留档，保留 10 份；当前二人长跑覆盖为每 1/100 update，保留 4 份归档 |

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

同日已按用户指令从 update 10 恢复正式二人主模型进程：`two_player + main + dead_rules + microbatch=24 + actor_batch=192 + anchor_wave=24 + temporal_cache_entries=576`，累计停止条件为 `3,000,000,000` 个分叉续局步。所有座位共享同一套 Policy/Layout，bootstrap 入口未运行。每个完成的 update 原子覆盖 `latest.pt`，每 100 update 归档一次并只保留最近 4 份；独立心跳每 15 秒采集资源状态。rev.15 完整 update 探针峰值 CUDA 分配 `16.66 GiB`，actor batch 与 learner microbatch 都未回退。

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

30 亿步作为二人“冷启动后扩大验证”里程碑是合理的：它约为冷启动分叉步预算的 `2.05` 倍，但只有正式主训练预算的 `25.7%`，所以不能被描述为顶尖棋力训练终点。rev.15 已启用物理页式 KV COW、单次动作 GPU→CPU 同步、packed 棋盘、长度分桶和规则静态查表；时间取决于端到端 Actor 吞吐，而不是 learner microbatch：

| 二人端到端吞吐 | 30 亿步纯运行时间 | 按 90% 可用率 | 结论 |
|---:|---:|---:|---|
| rev.15 main update 11 实测 651.97 步/s | 53.2 天 | 59.2 天 | 含 learner/检查点有效约 633 步/s；工程规划 61～70 天 |
| 1,000 步/s | 34.7 天 | 38.6 天 | 约 6 周，仍需实测原生 Actor/更快 GPU |
| 工程启动门槛 2,000 步/s | 17.4 天 | 19.3 天 | 达标后可以启动阶段训练 |
| 优化目标 3,000 步/s | 11.6 天 | 12.9 天 | 可在约两周内完成纯采样 |

rev.15 从 update 10 checkpoint 独立跑完一个 main update：`128` 个锚点、`1,024` 条续局、`313,935` 步；rollout `481.52 s`、**651.97 步/s**，完整 update `488.86 s`，普通原子 latest 保存约 `6～8 s`。`microbatch=24` 和 actor batch `192` 均未回退；页式 arena 峰值 `5,876 / 14,400` pages，CUDA 峰值分配 `16.66 GiB`。相对旧配置 update 3～9 的加权 `481.37` 步/s，吞吐提升 `35.4%`。之后使用 `metrics.jsonl` 至少 10 个新实现 update 的移动中位数持续更新 ETA：

$$
T_{days}=\frac{3\times10^9-N_{done}}{v_{plies/s}\times86400\times availability}.
$$

即使达到 2,000 步/s，正式排期也应按约 `3～4` 周墙钟预留。当前代码已经支持单模式 `torchrun/DDP`，并实现页式增量 KV 与分支前缀 copy-on-write；各 GPU 取得独立环境分片，只在 learner 同步梯度。继续达到该吞吐仍需要融合 paged-attention/CUDA Graph actor 或原生批量规则环境，并评估专用 actor GPU 上的一版本异步流水，不能把 DDP 卡数直接视为线性加速倍数。

#### bootstrap 二人模型的同口径实测

bootstrap 不是缩短 main 训练的无损替代品，而是约 `35.50M` 参数的阶段模型。RTX
4090 上使用 `actor_batch=128`、`anchor_wave=16`、`temporal_cache_entries=384` 和
`microbatch=24` 跑完一个正式 update：`1,024` 条续局共 `356,068` 步，rollout
`551.63 s`、**645.48 步/s**，完整 update `560.39 s`，检查点约 `1.4 s`；峰值
CUDA 分配 `3.82 GiB`、缓存池 `20.52 GiB`。`wave=8/actor=64` 的 600 步探针只有
`596.24` 步/s；`wave=16/actor=128` 为 `756.33` 步/s。继续增到
`wave=24/32` 会逼近 24GB 并明显退化，因此不用于 4090 长跑。

按首轮平均 `347.72` 步/rollout，30 亿步折算约 `8,425` updates / `8.63M`
rollouts。当前 4090 纯 rollout 为 `53.8` 个连续运行日，计入基础局、learner 和
每轮检查点为约 `54.8` 日；按 90% 可用率、策略分布变化和训练外评测，工程排期取
**60～70 天**。

RTX PRO 6000 尚未在本项目实测。根据 bootstrap 首轮中 Policy inference 约占
`81.2%`、Python 环境约占 `16.9%`，并结合 96GB 显存可安全起测
`actor_batch=256`、`anchor_wave=32`、KV cache `768`，保守估算为
`900～1,250` 步/s：30 亿步纯 rollout 约 `27.8～38.6` 天，工程排期约
**33～48 天**。这不是 NVIDIA benchmark，必须在目标卡完成一个正式 update 后
替换。启动入口为 `scripts/start_two_player_bootstrap.sh`。

### 7.3 RTX PRO 6000 Blackwell 96GB 时间规划

NVIDIA 官方规格中，[RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) 为 `24 GB GDDR6X、83 FP32 TFLOPS、1,321 AI TOPS`；[RTX PRO 6000 Blackwell Workstation Edition](https://www.nvidia.com/en-au/products/workstations/professional-desktop-gpus/rtx-pro-6000/) 为 `96 GB GDDR7 ECC、1,792 GB/s、125 FP32 TFLOPS、4,000 AI TOPS、600 W`。其中 AI TOPS 采用的精度/稀疏口径不能直接等同本项目 BF16 速度；较可比的 FP32 峰值只约 `1.51x`，显存带宽约为 RTX 4090 的 `1.78x`，而容量为 `4x`。

按二人最长上下文的实测显存斜率，96GB 卡建议 Policy `microbatch=64` 起步，稳定后测 `96`；全局 batch 仍为 128，所以从 4090 上每 epoch 约 6 个物理子批降至 2 个。Actor 建议从 `batch=384, wave=48, cache=1152` 起测，再测 `512/64/1536`。后一档页式 arena 约 `37.5 GiB`，仍能与模型、工作区和 learner 分阶段复用显存。增大这些批次主要降低调度开销，不改变 30 亿环境步目标。

rev.15 完整 update 中约 `80.6%` rollout 墙钟仍在 Policy inference、约 `17.4%` 是 Python 环境工作；因此 96GB 可继续扩大 actor wave，但不能按容量 `4x` 直接换算。以 4090 实测、FP32/带宽比例和 Amdahl 上限估算：

| RTX PRO 6000 方案 | 目标端到端吞吐（未实测） | 30 亿步墙钟规划（含 learner/评测） |
|---|---:|---:|
| 1× RTX PRO 6000 96GB | `900～1,200` 步/s | 约 **34～48 天** |
| 2× RTX PRO 6000 96GB，DDP 效率 85%～90% | `1,530～2,160` 步/s | 约 **19～29 天** |

两行都是根据 4090 实测瓶颈作的工程区间，不是目标卡 benchmark，并假定 600W Workstation Edition 与足够 CPU/散热。96GB 最大价值是能把更多页式 KV 和较大批量常驻显存，为后续优化创造条件，而不是自动把 Python 规则环境加速四倍。双卡若实际采用 300W Max-Q，应另做实测并预留额外降频余量。

### 7.4 RTX 4090 的 1～4 卡优化后时间规划

按 rev.15 单卡 rollout `651.97` 步/s、计入每轮 latest 保存后约 `633` 步/s外推剩余 `2,996,804,192` continuation plies。多卡 rollout 独立、learner 才做 DDP，但小尾波、无 NVLink、CPU 和 PCIe 拓扑会损失效率；2/3/4 卡分别按 `85%～90% / 80%～86% / 72%～80%` 估计。工程区间已计入 90% 可用率、分布漂移和评测余量：

| 配置 | 预计聚合吞吐 | 30 亿步纯 rollout | 工程墙钟规划 |
|---|---:|---:|---:|
| 1× RTX 4090 | 633 步/s（端到端实测外加平均保存） | 54.8 天 | **61～70 天** |
| 2× RTX 4090 | 约 1,076～1,139 步/s | 30.4～32.2 天 | **34～41 天** |
| 3× RTX 4090 | 约 1,519～1,633 步/s | 21.2～22.8 天 | **24～30 天** |
| 4× RTX 4090 | 约 1,823～2,026 步/s | 17.1～19.0 天 | **20～26 天** |

多卡吞吐是推算，必须在目标机器跑一次完整 update 复测。`anchor_batch=128` 不能
被 3 整除；三卡建议先用 `--anchor-batch 144 --microbatch 24`，每 rank 48 个锚点，
正好两波 `wave=24`。四卡若保留全局 128，可对比 `wave=16/batch=128/cache=384`
和 `wave=24/batch=192/cache=576`；24 个 CPU 逻辑核也可能成为约束。

同一 rev.15 负载的高端卡单卡规划如下。NVIDIA 官方规格为：H100 SXM `80GB /
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
`182.5` 步/rollout 对应约 `16.44M` rollouts 和 `16,054` updates；rev.15 update 11
实测 `306.58` 步/rollout，对应约 `9.79M` rollouts 和 `9,555` updates。仍先按
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
  --microbatch 24 --actor-batch 192 --rollout-anchor-wave 24 \
  --temporal-cache-entries 576 --target-continuation-plies 3000000000 \
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

四国默认只在每 5000 万环境步的评测前后保存完整 `latest.pt`，随后与此前最优旧模型比较 100 局；启动时把初始比较基线独立复制到 CPU 内存，首次评测时再落盘。平时、正常退出及异常退出均不额外保存。二人及直接使用 `--smoke-test` 的安装验收默认采用 `periodic` 策略，在进入 rollout 前写入 update 0，此后按周期及正常退出保存。可用 `--checkpoint-policy` 显式选择策略。

`latest.pt` 使用临时文件加原子替换，保存变体标记、共享的 Policy/Layout（四国另含 Critic）、对应优化器、阶段参考模型、update、累计锚点/续局/环境步、学习率缩放、待消费布局终局样本、所有 rank 尚未结束的基础对局及历史窗口，以及各 rank 的 Python/基础局/CPU/CUDA RNG。开启态还保存每位玩家的持久确定身份表和阵亡库存；关闭态没有阵亡历史张量。检查点不会为不同座位重复保存权重。PPO 或多卡发生异常时保留最后一个完整原子检查点，避免写入半轮更新的状态。

训练命令默认自动寻找对应模式、对应变体目录中的 `latest.pt` 并继续；不需要额外 `--resume` 参数。`--run-dir` 是基目录，程序自动追加变体名。`--no-resume` 只允许配合全新或空的运行目录使用；非空目录没有 `latest.pt` 时也会拒绝随机重启。确需新实验时应指定另一个 `--run-directory`。四国中断后从最近评测保存点恢复，可能重算两次评测间约 5000 万环境步；首次评测前尚无快照时只能在新目录重新开始。二人周期保存默认最多重算约 10 个 updates，当前二人任务覆盖为每 1 update 保存。恢复后中盘基础局从保存时的状态继续。评测中断会重试同一保存候选，已完成的评测不会重复，见[自动最优模型选择](best_model_selection_zh.md)。

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

当前版本是可运行、可反向传播、可中断恢复的 PyTorch 基线，已把候选动作和并行续局推理按 CUDA 子批执行，支持单机多 GPU DDP，并启用 autocast、TF32、fused SDPA 可用路径和 activation checkpoint。rev.15 还实现了物理页式 causal KV、页级 COW、单次动作 GPU→CPU 同步、packed 棋盘、learner 长度分桶及规则静态查表。要达到 2,000+ 步/s 的目标，还必须继续完成：

1. 将 Python 规则 actor 替换为 C++/Rust 或编译向量化环境，并与 Python 裁判持续差分；
2. 以融合 paged-attention/CUDA Graph 消除增量路径中剩余的 page gather 和小 kernel；
3. 在现有单机 DDP 上增加带一版本 behavior snapshot 的异步 actor-learner 服务及需要时的多节点/FSDP；
5. 增加训练外历史回归评测调度器、人类评测接口和长期断电演练。

30 亿步任务可以作为阶段训练启动，但不应把一次探针视为交付日期或把 30 亿步视为顶尖棋力保证；以首个完整 update 及其后 10 个 update 的移动中位数持续校准。
