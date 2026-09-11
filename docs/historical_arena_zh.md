# 历史模型棋力评测：公共约定

评测问题是：**新版本的 Policy + Layout 组合，在相同规则和推理预算下，是否比指定旧版本更强？**
不使用训练 loss 作棋力分数，不把同模型自对弈胜率或相关的 8 条训练续局当作独立比赛。

二人与四国的协议分别实现，只共用快照、安全加载、调度和统计基础设施：

| 模式 | 独立入口 | 对局协议 | 统计单位 | 文档 |
| --- | --- | --- | --- | --- |
| `two_player` | `evaluate_two_player` | 新旧模型各控制一方，两局换边 | 一对两局 | [二人说明](two_player_arena_zh.md) |
| `four_dark` | `evaluate_four_player --mode four_dark` | 新旧模型各控制整队，四局轮转 | 一组四局 | [四国说明](four_player_arena_zh.md) |
| `double_open` | `evaluate_four_player --mode double_open` | 同上，使用双明信息规则 | 一组四局 | [四国说明](four_player_arena_zh.md) |

三个模式的 baseline、训练 checkpoint、评测快照、对手池和输出目录均独立。四暗与双明即使模型尺寸相同也拒绝混用。
旧命令 `python -m junqi.training.evaluate_history` / `siguozero-eval-history` 仍仅用于二人。

## 训练与评测隔离

- 这是独立评测，不会启动、暂停、恢复训练，不会更新训练权重、优化器、规则或训练日志。
- 新旧模型使用各自 checkpoint 内的 Policy 与 Layout，冻结参数、`eval()`、无反传。评估的是二者的综合表现，不单独归因于 Policy。
- 裁判持有完整棋盘，行棋 Policy 只收到本座位 `GameHistory.state_for()` 产生的观察和合法动作。历史窗口随模型，四国的队友之间也不互传私有历史。
- 默认 Policy 温度 `1.0`、Layout 温度 `0.7`、总步数上限 `2000`；沿用规则引擎的 60 步无交互和棋与终局裁定，双方死规则必须一致。
- 每局运行到规则引擎结算。非法动作、加载失败、NaN/Inf 使评测失败，不能记成某模型输棋。保存种子、公开动作、座位与最终奖励。
- checkpoint 通过 CPU/mmap 只读加载；只复制推理权重，优化器与 reference 不进入加速器。每个 rank 中候选和对手各加载一份 Policy + Layout，并发对局共享所属版本的权重。每模型默认至少 8 个 temporal cache entries、至少 2048 个棋盘缓存条目，缓存容量自动随并发局数扩展，对局结束后释放其历史引用。
- 只加载可信的本地产生的 PyTorch checkpoint，不接收来历不明的 pickle。

## 周期与对手池

- 新套件必须指定较早的 `--baseline`，生成独立的带 SHA-256 推理快照；它不会随 `latest.pt` 改变。
- 默认每增加 **25 updates** 触发一轮。对手为 **固定 baseline + 最近 2 个已评测版本**；二人每个对手默认 400 局，四国默认 800 局。
- `--watch` 每 60 秒检查已发布的 manifest；`--once` 执行一轮符合间隔的评测，或恢复未完成轮次，然后退出，适合外部作业调度器。
- 首轮只要 latest 距 baseline 达到间隔便立即评测；没有新 checkpoint 时不会重赛，不会恢复已暂停训练。
- 评测慢于训练时，先完成被冻结的候选版本，再取符合间隔的最新版本，不积压补评所有中间 update。
- 开赛前写入 `state.json`。中断后已完成 matchup 复用结果；未完成 matchup 按保存的种子重新开始，不重复累计。
- 输出目录用 OS 文件锁阻止重复 evaluator，退出自动解锁。不要手动删除 `.arena.lock`；共享存储必须支持跨节点 advisory file lock。
- 只保留 baseline、最近的候选推理快照及未完成任务依赖；清理仅针对评测生成的副本，训练 checkpoint 从不删除。结果与公开棋谱持续保留，需要长期重演的旧权重请另行归档。
- 推理快照不能恢复训练；每个 main 副本约一套 Policy/Layout 权重大小，另外要给候选快照、临时文件及持续增长的棋谱留出空间。

套件固定模式、参数、规则/编码器/模型/评测实现源码指纹及 Torch 版本，rank 间不一致会拒绝运行。
**并行评测使用 `arena_version=3`，单局采样改为独立随机数流；已有 v1/v2 历史评测输出不迁移、不覆盖，必须使用新输出目录。**
修改评测代码、baseline 或参数后同样需要新目录，不能把前后成绩当成一个不变实验直接合并。

## 如何判定提升

`score = (wins + 0.5 * draws) / games`，50% 表示该 matchup 打平；四国的每局胜负指整队最终胜负，不是某个座位是否存活。

统计样本是**完整的两局换边对 / 四局轮转组**，不能把同组比赛或四国的两位队友当成独立样本：

- `bootstrap_ci95_descriptive`：对整组平均分重采样的 95% 区间，仅作描述。小样本全胜时可能退化为 `[1, 1]`，不能直接用来宣布提升。
- `score_ci`：用于判定的保守 Hoeffding 区间。每组平均得分在 `[0, 1]`，半径 `sqrt(log(2/alpha)/(2*N))`，`N` 是两局对数或四局组数，而非总局数。
- 第 `r` 轮有 `m` 个对手时，每个 matchup 使用 `alpha = 0.05 / (r*(r+1)*m)`，同一模式的同一套件累计支出不超过 0.05。这不是跨三个独立套件联合的 95% 保证。
- 每轮分配新的、不重叠种子块；断点续评用保存的种子。区间基于冻结模型条件下各组是独立随机样本的常规模拟假设，不覆盖规则系统性错误或评测集泄漏。

| 结果 | 含义 |
| --- | --- |
| `ahead_of_opponent` | 修正区间下界 > 50%，有领先该旧版本的证据 |
| `behind_opponent` | 修正区间上界 < 50%，有落后该旧版本的证据 |
| `inconclusive` | 区间跨过 50%，证据不足；不是“没有学会” |
| `improvement_against_tested_opponents` | 本轮对每个被测试旧版本都有领先证据 |
| `regression_against_historical_opponent` | 本轮至少落后一个历史对手，检查退化或非传递性 |
| `smoke_test_not_strength_evidence` | 仅工程冒烟，不输出棋力结论 |

默认 200 个组样本适合周期观察；首轮单对手的判定区间半径约 10.47 个百分点，53% 得分不会被宣布为显著提升。
微小提升可在预先确定的新套件使用 1000～5000 个组样本和未使用的种子；不能看到结果后反复扩样、筛选种子或重建套件只保留有利结论。
同时查看座位/队伍分项得分及和棋原因。最大步数和棋 >10%、总和棋 >80% 会提示，但不代表规则必然有错。
这里只能得出相对于这些历史对手的结论，不是人类 Elo，也不能排除对未测试策略退化。

## CPU / CUDA / NPU 与多进程

各自命令见模式说明。默认 `--device cpu --cpu-threads 1`，WSL 使用 `siguozero` 环境，纯 CPU 设置 `CUDA_VISIBLE_DEVICES=''`。
main 模型 CPU 正式比赛可能很慢。CUDA/NPU 应由调度器分配独立或空闲评测卡，不能直接挤占满载训练卡；CPU、内存和磁盘仍有开销。

NPU 独立脚本为 `scripts/evaluate_two_player_npu.sh` 和 `scripts/evaluate_four_player_npu.sh`；默认 1 个评测进程，用 `NPROC_PER_NODE` 显式增加。
公共 `scripts/evaluate_npu_cluster.sh` 未设置 `EVAL_MODE=history` 时仍为原来的同模型自对弈评测，不具备跨版本比较含义。
CANN、`NNODES`、`NODE_RANK`、`MASTER_ADDR` 与原集群脚本一致，评测 rendezvous 端口必须与训练区分。
多节点每个节点启动命令，设置不同 `NODE_RANK`；训练与输出目录必须是真正共享存储，不能只是同名的节点本地目录。

多 rank 按整组 index 分片，绝不拆开两局对或四局组。每个 rank 同时只持有新旧两个版本，不一次载入整个对手池。
rank 内默认并发 32 局，同一版本的待行棋请求合成最多 32 条的推理批次，CPU 环境转移默认使用 4 个 worker。短局结束后立即补入下一局，不等待同批长局全部结束；多 rank 仍按完整组分片，全局总局数保持不变。
可用 `--parallel-games`、`--inference-batch-size` 和 `--environment-workers` 调整各 rank 的资源用量，NPU 脚本对应 `EVAL_PARALLEL_GAMES`、`EVAL_INFERENCE_BATCH_SIZE`、`EVAL_ENVIRONMENT_WORKERS`。详情见[并行评测与资源调优](parallel_arena_zh.md)。
种子不依赖 world size；每局使用独立随机数流，完成顺序不会消耗其他对局的采样随机数，但跨批量、硬件、精度或算子版本的浮点数差异仍可能改变边界处的采样结果。
目前的自动验证不替代实体 Ascend NPU 或真实 main 长局验收。

## 输出与验收

- `report.md`：带模式名称的中文历史成绩表，固定 baseline 行便于观察长期趋势。
- `latest.json` / `history.json`：最新一轮与全部历史。需结合 `status.json`；失败时 latest 仍可能是上次成功结果。
- `status.json`：包含模式、等待/评测/完成/失败状态；开始时间不是每 rank 心跳。实时进度看控制台 `arena_groups_completed`、`arena_groups_target`、`games_per_group`。
- `rounds/roundNNNNN.json`：候选、对手池、哈希、判定与套件配置。
- `matches/*.json`：matchup 详细统计。二人为 `pairs` / `seat_scores`；四国为 `rotation_groups` / `team_scores` / `rotation_scores`。
- `matches/*.<attempt>.rankNNN.jsonl`：逐局审计。仅成功 matchup JSON 的 `game_shards` 列表权威，其他 attempt / `.tmp` 不纳入成绩。

内部共用的 `settings.pairs` 字段为兼容历史 API，表示组样本数；四国 CLI 明确用 `--groups`、结果明确用 `rotation_groups`。读取报表时应同时检查 `mode`、`statistical_unit` 和 `games_per_group`，不要硬编码“组数 × 2”。

工程测试使用 tiny 模型与短局，显式 `--smoke-test`，覆盖规则结算、信息隔离、队伍轮转、冻结权重、统计、防重复、断点续评、快照保护及 CPU/Gloo 分片，不证明真实 main 棋力提升。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```
