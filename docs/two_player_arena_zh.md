# 二人军棋：历史模型棋力评测

独立实现：`arena_two_player.py`；入口：`python -m junqi.training.evaluate_two_player`，安装后亦可用 `siguozero-eval-two-player`。
仅接受 `two_player` checkpoint；不接受四暗、双明或 `--groups`。共用安全、统计与输出约定见[公共说明](historical_arena_zh.md)。

## 对局与计分

每个样本中新旧 Layout 各采一份合法布阵，固定这两份布阵完成两局：

| 局 | 座位 0（先手） | 座位 1 |
| --- | --- | --- |
| 1 | 新模型及其布阵 | 旧模型及其布阵 |
| 2 | 旧模型及其布阵 | 新模型及其布阵 |

每方只读自己的观察。胜/和/负按新模型的单局最终奖励记为 1/0/-1，得分为 1/0.5/0。
两局平均分构成一个统计样本；`--pairs 200` 表示 200 对、400 局，而不是 200 局。
默认每 25 updates 一轮，固定 baseline + 最近 2 个已评测候选，最多 3 个对手、1200 局/轮。
`seat_scores` 分别列出新模型在两个座位的平均得分，`statistical_unit=two_game_seat_pair`。

## 周期运行

下面是新 v2 套件示例。请先确认 baseline 是真实存在的旧 checkpoint，命令只启动评测，不恢复训练：

```bash
CUDA_VISIBLE_DEVICES='' python -m junqi.training.evaluate_two_player \
  --checkpoint-dir runs_optimized_v14/two_player/with_dead_rules/checkpoints \
  --baseline runs_optimized_v14/two_player/with_dead_rules/checkpoints/update_000000100.pt \
  --output-dir eval_history/two_player_main_v2 \
  --device cpu --cpu-threads 1 \
  --every-updates 25 --recent-opponents 2 --pairs 200 \
  --watch --poll-seconds 60
```

外部调度器使用 `--once` 替换 `--watch --poll-seconds 60`，后续恢复同一套件时可省略 `--baseline`。
固定模型哈希与种子，未完成 matchup 可重试，已完成结果不会重复记分。
若只是验证工程链路，在全新目录使用 `--pairs 1 --max-plies 4 --smoke-test --once`；这不提供棋力提升证据。

## Ascend NPU

```bash
NPROC_PER_NODE=2 MASTER_PORT=29511 \
EVAL_OUTPUT_DIR=/mnt/shared/eval_history/two_player_main_v2 \
EVAL_BASELINE=/mnt/shared/runs/two_player/with_dead_rules/checkpoints/update_000000100.pt \
EVAL_PAIRS=200 EVAL_EVERY_UPDATES=25 EVAL_RECENT_OPPONENTS=2 \
  bash scripts/evaluate_two_player_npu.sh \
  /mnt/shared/runs/two_player/with_dead_rules/checkpoints/latest.pt --watch
```

必须先分配独立/空闲评测卡；默认仅 1 个进程，示例显式使用 2 个。二人使用 `EVAL_PAIRS`，误设 `EVAL_GROUPS` 会报错。
该脚本固定二人模式；四国请使用[四国专用入口](four_player_arena_zh.md)。
