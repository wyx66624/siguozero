# 训练中的自动最优模型选择

三个训练模式默认从训练进度 **30%** 开始评测，此后每增加 **5 个百分点**评测一次：
`30%、35%、40%、…、100%`。每轮让当前 `Policy + Layout` 与此前选出的最优模型
对弈 **1,000 局**，按候选模型的胜、和、负决定是否更新最优模型。

## 进度与比赛规则

训练同时受到更新次数和可选环境步数预算限制，先达到任一预算就停止。因此评测使用
两种完成比例中的较大值：

```text
进度 = max(已完成 updates / total_updates,
           已完成 step_budget_counter / step_budget_target)
```

未设置环境步数目标时只使用 updates；`target_environment_plies` 使用累计训练环境交互，
包含采样对局和实际模拟分支。二人 GRPO 只有使用旧 `target_continuation_plies` 时才仅计续局。
独立选优比赛的交互单列，不写入训练样本预算。比赛在一轮训练更新完成后触发，不会把尚未更新完的参数
送入评测。如果一次更新跨过多个门槛，只用这次实际保存的参数评测一次，归到最近的
已跨过门槛，并记录跳过的门槛；不会虚构中间进度对应的历史模型。

首次启动时以初始参数作为基线，这份参数尚未经过对弈验证；初始 `best.pt` 只是
后续比较的起点。带自动选择记录的正常续训恢复已有最优模型和评测进度。
旧运行第一次启用此功能时，以本次恢复的参数作为未评测基线，从下一个尚未经过的
门槛开始比较，无法补评已经不存在的历史参数。

二人模式每 2 局组成一组换边比赛，共 500 组；四暗和双明每 4 局组成一组整队轮转
比赛，共 250 组。四国由一个版本控制一支队伍的两个座位，按整队胜负计分。
多卡按完整比赛组分配工作，**全体 rank 总计 1,000 局**。不同轮次使用确定且分离的
随机种子。单局最多 2,000 步，达到上限按和棋计入得分，并保留和棋原因。

评测默认在每个 rank 同时推进 **32 局**，按模型版本合并推理请求，每批最多 **32 条**，
并使用 **4 个 CPU 环境 worker**。短局结束后立即补入下一局，长局不会拖住整批比赛；
候选和最优版本各自只加载一份 Policy + Layout 权重。单局有独立随机数流，组级换边、
整队轮转和统计口径保持不变。并行只改变资源调度，**不会把 1,000 局乘以并发数或卡数**。
缓存容量自动随并发局数扩展，配置及资源调优见[并行评测](parallel_arena_zh.md)。

```text
候选得分率 = (候选胜局数 + 0.5 × 和棋局数) / 总局数
候选得分率 > 50%：选择当前模型作为新的最优模型
候选得分率 ≤ 50%：保留此前最优模型
```

即默认 1,000 局时，候选累计得分必须严格高于 500 分；相等时不替换。选择评估的是
`Policy + Layout` 联合对弈能力，四国 Critic 不参与比赛。评测还保存胜负和分布及
统计区间，可查看相邻轮次的变化。统计以整组换边或轮转比赛为单位，使用 Hoeffding
区间，并将整个评测计划的 `0.05` 错误预算按门槛数分配；默认 15 个门槛，每轮为
`0.05 / 15`。另提供描述性的组级 bootstrap 区间。

超过 50% 是自动选择规则，与统计区间分开判断；有限样本、随机布局和对手变化意味着
一次晋升并不能证明棋力在统计上显著上涨。跨轮次也不能只看得分率：
每轮的对手可能已经更强，进一步验证可使用固定对手的
[历史模型评测](historical_arena_zh.md)。

## 结果与继续训练

结果保存在各模式、死规则变体对应的运行目录，例如
`runs/two_player/with_dead_rules/`：

| 路径 | 用途 |
| --- | --- |
| `checkpoints/best.pt` | 当前选出的最优 `Policy + Layout` 推理权重 |
| `checkpoints/latest.pt` | 最新训练检查点，包含优化器等完整续训状态 |
| `model_selection/state.json` | 已评测进度及当前最优模型记录 |
| `model_selection/round_030.json` 等 | 每个评测门槛的比赛汇总及选择结果 |
| `model_selection/history.jsonl` | 根据已提交的轮次结果原子重建的选择历史，用于查看棋力变化 |
| `model_selection/snapshots/` | 选择过程使用的独立推理权重快照 |

**训练继续优化当前参数，最优模型单独保存。** 候选落败时保留 `best.pt`，不会回滚
正在训练的模型、优化器、Critic、学习率或训练计数。这样既能持续探索，又能始终用
当前选出的最优版本进行推理。`best.pt` 是推理快照，正常断点续训仍应使用完整的
`latest.pt`，不要将二者互换。

已有选择记录的运行续训时，必须保持 `total_updates`、环境预算目标及其计数口径
以及评测的起点、间隔、局数、最大步数、种子、配置中的缓存条目下限一致。改变这些值会改变
评测进度或比赛条件，程序会拒绝复用原选择记录。需要调整时，可用新运行目录并通过
`--init-from` 载入权重；如果只是继续训练而不再自动评测，可加 `--no-model-selection`。
旧 PPO 选择记录虽然使用 `target_continuation_plies` 字段名，实际已经统计总环境交互；
升级时自动迁移字段，保留原最优模型和评测进度。并发局数、推理批量和环境 worker 数属于执行资源参数，可在继续训练时调整；
它们不改变评测门槛、总局数或最优模型选择规则。

使用最优二人模型推理示例：

```bash
python -m junqi.training.infer_two_player \
  --checkpoint runs/two_player/with_dead_rules/checkpoints/best.pt
```

## 配置与启动

[共享 YAML](../configs/bootstrap.yaml) 的顶层 `model_selection` 配置默认为：

```yaml
model_selection:
  enabled: true
  start_percent: 30
  interval_percent: 5
  games: 1000
  max_plies: 2000
  seed: 20260910
  temporal_cache_entries: 8
  parallel_games: 32
  inference_batch_size: 32
  environment_workers: 4
```

`start_percent` 和 `interval_percent` 必须是 1 到 100 的整数；局数必须为正整数，
二人模式须能被 2 整除，四国模式须能被 4 整除。最大步数、缓存条目数、并发局数、
推理批量和环境 worker 数必须为正整数。
种子及全部评测使用的种子范围必须可由非负有符号 64 位整数表示。

训练入口直接使用上述默认值，也可通过 CLI 覆盖：

```bash
python -m junqi.training.train_two_player --model-scale main \
  --arena-start-percent 30 --arena-interval-percent 5 \
  --arena-games 1000 --arena-max-plies 2000 \
  --arena-parallel-games 32 --arena-inference-batch 32 \
  --arena-environment-workers 4
```

`--no-model-selection` 可关闭自动比赛；`--smoke-test` 默认关闭比赛，以保持安装验收
快速完成。程序调用 `TrainingSettings.from_yaml(..., tiny=True, overrides={...})`
时可显式覆盖 `arena_enabled=True`，供小规模集成验证使用。比赛需要额外算力，期间
暂停训练更新；比赛的步数不计入训练预算。
