# 四国军棋 PPO、独立价值模型与 RTX 4090 显存实测

后半程独立的 20% 冻结对手与 20% 冻结队友、各席位学习归属和内存优化见[历史对手、队友训练与评测](historical_opponents_zh.md)。

revision 24：和棋训练奖励为 **-0.15**，动作输入增为六维；Critic 新增共同和棋惩罚
价值头，和棋分量不随下一玩家队伍翻转。当前实现与格式 6/7 → 8 迁移说明见
[和棋负奖励与步骤倒计时](draw_penalty_countdown_zh.md)。以下参数和计时为标注版本的历史记录。

2026-09-11 补充：数组历史、批量动作损失、固定席位 KV 及 CUDA Graph 已默认启用，
模型结构及检查点格式保持兼容。同检查点 4090 完整训练轮从 56.05 秒降至 19.16 秒，
详见[数据路径优化与验证](ppo_pipeline_optimization_zh.md)。下文较早计时保留为历史基线。

当前四国 main 使用 48 局、12288 步采样批、32 条历史的学习微批和 4 个 CPU 环境进程，
启用直接因果 SDPA、关闭激活重算。当前工期与满窗口容量见
[并行环境与学习端优化](ppo_parallel_optimization_zh.md)。此前 237 步/秒的
[正式批量复测](four_dark_optimized_training_eta_zh.md)保留为历史对照。

日期：2026-09-11；当前配置 revision 22。适用于 `four_dark` 和 `double_open`。
这两个模式的行棋训练现已使用 PPO；`two_player` 保留 K=4、M=2 的 Game-GRPO。
整盘单层编码与本次验证见[整盘向量投影](whole_board_linear_zh.md)。
当前默认并行数、参数和训练时间见[128＋128 并行训练核算](compact128_parallel_training_eta_zh.md)。
30 亿环境交互步预算和旧版端到端计时见
[序列训练优化与预算报告](four_player_ppo_optimization_zh.md)。

## 1. 参数量与模型职责

以下为 revision 22 默认开启死规则时，按实际 `nn.Module.parameters()` 统计的参数量。
四暗和双明使用相同架构，不按四个座位复制模型。

| 规格 | 行棋 Policy | 独立 Critic | 布阵 Layout | PPO 可训练参数合计 |
|---|---:|---:|---:|---:|
| bootstrap | 11,109,250 | 10,993,921 | 8,887,296 | 30,990,467 |
| main | 36,324,226 | 36,208,897 | 17,292,288 | 89,825,411 |
| extended | 53,134,210 | 53,018,881 | 25,697,280 | 131,850,371 |

`WholeBoardEncoder` 把按点位排列的整盘类别向量、模式和可选阵亡位拼成一个固定输入，
只经过一个 `Linear(18009, 128)`（关闭死规则为 `Linear(17934, 128)`）。不创建逐点
神经特征或棋盘 Transformer。其输出与 128 维动作向量 concat 成 256 维时序 token。
main 的时序网络仍为 32 层、8 个注意力头，FFN 为 1024；动作头直接预测起点及给定起点后的终点 logits。
动作输入为[五维坐标单层投影](action_linear_zh.md)：起点 x/y、终点 x/y、相对行动方
直接经过 `Linear(5,128)`，不包含战斗结果字段或字段 Embedding。

`GameValueTransformer` 使用同样的整盘线性编码、动作编码与时序网络，用
`Linear(256, 1)` 替代动作头。它输出该观察者队伍的预期终局回报，不接收裁判隐藏身份。
骨干从 Policy 复制初值；两个模型各自拥有参数、优化器和推理缓存。

四国训练常驻 Policy、Critic、Layout，以及一份冻结 Layout 参考模型。
main 可训练参数合计 **89,825,411**，加冻结 Layout 共 **107,117,699**。
推理和历史评测只加载 Policy + Layout，不加载 Critic 或优化器。

宽度覆盖在 `models.four_player` 中，仅作用于非 tiny 的四国训练。布阵模型、层数、
1000 步历史和二人模型宽度保留原值。旧 512 维检查点不能直接恢复为新 256 维模型；
正式入口使用独立根目录 `runs_four_player_ppo_128_3b`，旧检查点保留。

## 2. 单条真实轨迹替代 8 条终局续局

每个 update 按以下过程执行：

1. 冻结本轮 Policy/Critic 参数，每局每次采样一个合法动作并真正执行。
2. 保存执行前的玩家可见历史、动作、旧 log-prob、旧价值及真实奖励。
3. 达到全局 transition batch 后停止采集，未结束的基础对局保留到下一轮。
4. 对未终局的采集边界，用冻结 Critic 估计下一状态价值；真正终局的后继价值为零。
5. 使用 GAE 计算优势和价值目标；跨 rank 标准化优势，但保持价值目标原尺度。
6. 清空两套推理缓存，按每 rank 512 个决策组织优化器小批，先更新 Policy，
   再更新 Critic。双方使用同一批冻结目标；两个网络的反传图不会同时驻留。

四国是两队对抗，队伍按 `game.team_of(player)` 判断。令 `sigma_t` 在下一行动者
与当前行动者同队时为 +1，异队时为 -1；终局令 `d_t=0`，其他情况为 1：

```text
deltaZ = rZ + gamma * d * sigma * Z_next - Z
deltaC = rC + gamma * d * C_next - C
AZ = deltaZ + gamma * lambda * d * sigma * AZ_next
AC = deltaC + gamma * lambda * d * AC_next
A = AZ + AC; V = Z + C
```

这样可以处理玩家出局后跳过座位的情况，不能机械地每步乘 -1。终局奖励为
胜 +1、和 -0.15、负 -1，仍没有人为中间奖励。胜负分量为 Z，共同和棋惩罚分量为 C，
两者各自计算 clipped value loss，使用同一个 Critic 编码器前向。采集批次结束不算终局；配置的
`max_game_plies` 仍按现有裁判规则判为和棋。

Policy 使用 PPO clipped objective 和熵项，以相对采样旧策略的近似 KL 提前停止。
Critic 使用 PPO clipped value loss。默认 `gamma=1`、`lambda=0.95`，策略与价值
clip 都是 0.2，value coefficient 为 0.5；两者学习率默认 1e-4，分别最多/固定
3 个 epoch。每个 epoch 内有多个优化器小批，各小批由 microbatch 梯度累积
组成；扩大采样批不会同步降低优化器更新频率。Critic 不参与 Policy 的反传，
价值目标也不会随着本轮 Critic 更新而重算。

本地 4090 覆盖上述策略裁剪默认值，启用[阶段与优势自适应上界](adaptive_clipping_and_resume_zh.md)，
在预算 1/3、1/2、3/4 等节点间平滑收窄。Critic 的 0.2 value clip 保留原值。

**本次替换的是行棋优势估计。** 独立的 25 步布阵模型继续根据完整对局的真实
终局结果做已有的 clipped terminal-return 更新；布阵训练没有新增模拟分支。
它的 Layout GRPO、Layout reference 和缓冲区机制保持独立。

## 3. 修改前 RTX 4090 24GB 实测与并行设置

以下计时和容量记录来自旧空间编码器。revision 20 的测试见[整盘投影验证](whole_board_linear_zh.md)，不能直接沿用旧数字预测新结构工期。

环境：本机 WSL2、RTX 4090 24GB、PyTorch `2.11.0+cu130`、BF16 autocast、
FP32 权重/AdamW、activation checkpointing、main、死规则开启、1001 token。
三个可训练模型的 AdamW 状态均已分配，Layout reference 同时驻留。
探针使用每个样本、每个 token 都不同的合成棋盘历史，避免重复状态去重导致
虚假的大 batch 测量。这是容量压力测试，不是合法棋谱或棋力测试。

| 阶段/配置 | CUDA allocated 峰值 | CUDA reserved 峰值 |
|---|---:|---:|
| microbatch=8，Policy 与 Critic 分阶段反传，连续两次更新 | 14.03 GiB | 14.86 GiB |
| 8 局，actor batch=8，两模型各 96 个 KV 条目，32 个满长玩家历史 | 9.75 GiB | 10.00 GiB |
| 16 局，actor batch=16，两模型各 192 个 KV 条目，64 个满长玩家历史 | 15.94 GiB | 16.43 GiB |
| 序列 microbatch=8，每序列 64 个决策，Policy 反传 | 15.50 GiB | 17.97 GiB |
| 20 局，actor batch=20，两模型各 240 个 KV 条目，80 个满长玩家历史 | 19.02 GiB | 19.70 GiB |

没有推理缓存时，模型与全部优化器状态的常驻分配约 3.54 GiB。
原始结果见 [4090 PPO 容量数据](benchmarks/2026-09-10_ppo_4090.json)。
后两行来自新增的[序列训练及 20 局容量数据](benchmarks/2026-09-10_ppo_sequences_20_4090.json)。

**以下是旧结构当时的 20 局并行配置。** revision 22 的并行配置以[新版实测](compact128_parallel_training_eta_zh.md)为准：

```text
model_scale=main
policy_microbatch=8       # 旧结构：每次反传最多 8 条具有相同前缀的历史序列
max_samples_per_sequence=64
optimizer_minibatch_samples=512 # 每 rank 每次优化器 step 的决策数
base_game_pool_size=20    # 独立对局数
actor_inference_batch=20  # 单次推理状态数
temporal_cache_entries=240 # 每个模型独立设置，不是两个模型合计
transition_batch=5120     # 每轮全局真实对局动作数
target_environment_plies=3000000000 # 所有对局、座位合计的真实动作数
```

如需降低到 16 局，三个并行参数一起改为 `16 / 16 / 192`。每个模型的缓存按
`3 × 4 个座位 × 对局数` 留容量。8 局两模型 KV 合计 4.69 GiB，16 局为
9.38 GiB；32 局的 KV 单项就约 18.75 GiB，加模型、优化器和临时工作区会非常紧张，
本次没有把 32 局作为可用配置。`microbatch` 控制反传批，不能单独限制 rollout
的并行环境和缓存。两阶段必须清空缓存后才进行反传。

这些峰值针对本次软件环境和压力样本；Windows 显示程序另占部分显存。
单卡反传 OOM 会将尚未执行 optimizer step 的小批按 `8 -> 4 -> 2 -> 1` 重试。
多卡发生 OOM 时保留最后完整检查点并退出，所有 rank 以更小批次一起重启。

## 4. 运行、迁移和评测

架构切换后使用新的 run-dir。下方 `--init-from` 只适用于同为 v6、编码和模型维度均相同的 GRPO→PPO 权重初始化，不会把旧图编码器或 512 维时序权重转换为新 256 维结构。

RTX 4090 正式规格启动命令（不会自动启动训练）：

```bash
bash scripts/start_four_player_ppo.sh four_dark
bash scripts/start_four_player_ppo.sh double_open

# 四张卡，每张卡保留所选默认并行工作量
NUM_GPUS=4 \
  bash scripts/start_four_player_ppo.sh four_dark
```

脚本默认使用 `/root/anaconda3/envs/siguozero/bin/python`，可通过 `PYTHON_BIN`
覆盖。输出进入 `runs_four_player_ppo_128_3b/<mode>/<dead_rule_variant>`。
当前本地配置每 1000 万环境步保存完整续训状态，正常停止时也保存。前半程每 5000 万步
与此前最优版本对弈 100 局；从 15 亿步起取消冠军比较，仍每 5000 万步评测固定历史对手集。
评测前后保存完整状态，并保留该轮推理快照；评测结果不阻止继续使用最新模型。保存与比较规则详见
[自动最优模型选择](best_model_selection_zh.md)。
`--transition-batch` 是 `--anchor-batch` 的别名，PPO 的单位是实际动作；
`--target-environment-plies` 设置累计训练环境交互目标，含采样对局和实际模拟分支，
例如 8 条分支各走 32 步合计 256。详见[统一定义](environment_step_budget_zh.md)。
当前 PPO 不执行额外蒙特卡洛分支，所以总交互数暂时等于采集的对局动作数。
使用同一种环境交互单位，也不能保证不同算法获得相同棋力或产生相同算力消耗。
配置和启动脚本均默认 30 亿环境交互步，外层更新数为目标除以全局采样批向上取整，
最后一批按剩余步数收集；默认 warmup 2000 次。PPO 会根据目标及采样批自动推导
更新上限和学习率周期，显式 `--updates` 优先；冒烟测试仍默认一轮。
此目标不保证棋力收敛，旧 GRPO 等价换算仅保留为显式兼容选项。

当前检查点使用 v6 格式，拒绝直接加载旧编码器的 v4/v5 权重。相同新架构的 PPO 检查点包含 `algorithm=ppo`、
`critic`、`critic_optimizer`、critic GradScaler，以及实际环境步计数。
缺少 `algorithm` 的同架构 v6 检查点按 GRPO 处理，跨算法续训仍被拒绝；旧 v4/v5 在格式检查时即被拒绝。

从既有四国 GRPO 权重初始化时，必须明确使用一个新输出目录：

```bash
bash scripts/start_four_player_ppo.sh four_dark \
  --init-from /path/to/old/four_dark/with_dead_rules/checkpoints/latest.pt \
  --run-dir runs_four_player_ppo_migrated
```

迁移保留 Policy/Layout 权重，以 Policy 骨干初始化 Critic，重新开始优化器和
计数；源检查点不变。PPO 的正常断点恢复会恢复全部三个优化器、价值模型、
未结束对局、历史和 RNG。Policy 已更新但 Critic 更新失败时，不把半轮状态写入
`latest.pt`，保留最后一个完整原子检查点。

`train_npu_cluster.sh` 的四国入口也切换为 PPO，并使用独立的
`runs_npu_910b_ppo_128_3b` 默认目录，也按环境交互目标推导更新数；以上 4090 数字
不代表 NPU 显存/吞吐实测。
CPU 双进程 DDP 已验证；NPU 仍需实机验收。

关键日志为 `loss/policy_ppo`、`loss/critic_total`、`critic/value_mse`、
`policy/approx_kl_old`、`rollout/policy_samples`、`cumulative/environment_plies`、
`timing/policy_backward_seconds` 和 `timing/critic_backward_seconds`。
PPO 的 `rollout/root_candidates` 与 `rollout/terminal_continuations` 应为零。

## 5. 训练时间与验证边界

当前结果见[revision 21 实测与各卡预算](action_linear_training_eta_zh.md)。
以下保留逐点棋盘编码阶段的历史对照，不能用于当前架构排期：
[序列训练优化与预算报告](four_player_ppo_optimization_zh.md)中，
同一批真实样本的历史反传约快 10.4 倍。但新预算是 30 亿环境交互步，按短基准
外推训练本体约需 2.5～4.2 年连续运行，80% 可用率约 3.1～5.2 个日历年；
本地自动选优只在前半程进行，共 29 轮、2900 局；从 15 亿步起为 31 轮固定历史对手评测，
6 个对手时每轮 2400 局。此前只按冠军比较局数估计的总工期不包含这部分新增评测成本，
需按实际完整周期重新估算。并行资源条件见[时间重估](four_player_ppo_eta_budget_zh.md)。
这些不是长期实测或上界。满长窗口大量滑动时，
前缀复用比例可能降低，需要用实际训练曲线修正。此前独立历史样本路径的
[时间报告](four_player_ppo_timing_zh.md)仅保留为历史基线。

同样采集 B 个训练决策，原行棋 GRPO 约需 `8 × B × 平均剩余局长` 次分叉动作，
另有 B 次基础对局动作；PPO 仅需 B 次基础对局动作，加价值推理/反传。
例如 B=128、平均剩余局长=334.5，原分叉部分约 342,528 步，PPO 只执行 128 个
真实动作。这个比值说明取消了多少模拟，**不等于端到端加速倍数**。

新增满长序列探针中，8 条不同历史、每条 64 个决策位置的 main 反传，Policy
约 3.72 秒、Critic 约 3.67 秒；相同数量的 learner 样本会多一套反传。
PPO 还引入价值估计偏差和自对弈非平稳性，原来的收敛步数/棋力保证不能沿用。
能确认的是实现、容量和训练闭环可行；达到同等棋力所需的总时间仍须用历史
对手固定种子评测来判断，不能宣称简单加速 8 倍。

测试覆盖优势符号、终局/截断边界、真实单轨迹采集、PPO 裁剪、两个网络各自学习、
Critic/优化器恢复、显式旧权重迁移及半轮失败保留完整检查点。另已执行 CUDA
训练闭环、CPU 双进程 DDP，以及真实 main 的 1001-token 容量探针。

最新 WSL 全量回归为 `174 passed, 2 skipped, 172 subtests passed`，覆盖前缀
合并与独立计算的输出、所有参数梯度、优化器小批覆盖、30 亿环境交互步停止及
学习率周期、短目标实际收尾和兼容预算。四暗 CPU
双进程 DDP 完成两轮更新、实际终局、布局更新和检查点保存；main CUDA 完成
16 局两轮及 20 局一轮完整训练与满长容量探针。此前双明 tiny CUDA 冒烟从
update 4 恢复到 update 5，累计 80 个真实动作，根候选与终局分叉均为零；
双明本轮没有同等规模的 main 长期计时。

全量回归在 WSL 训练环境执行；Windows 原生回归中的历史评测测试存在 `.pt`
文件占用和默认 GBK 读取 UTF-8 报告的问题。这些评测模块不属于本次 PPO 改动，
不将 WSL 通过表述为所有 Windows 评测也已通过。

算法依据：[PPO 论文](https://arxiv.org/abs/1707.06347) 与
[GAE 论文](https://arxiv.org/abs/1506.02438)。队伍价值符号、裁判终局和共享玩家
模型是本仓库针对四国军棋的实现约定。
