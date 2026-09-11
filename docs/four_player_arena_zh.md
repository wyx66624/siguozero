# 四国军棋：历史模型整队棋力评测

独立实现：`arena_four_player.py`；入口：`python -m junqi.training.evaluate_four_player`，安装后亦可用 `siguozero-eval-four-player`。
模式为 `--mode four_dark`（默认四暗）或 `--mode double_open`（双明）。二者的 checkpoint、baseline、对手池、输出目录完全分离；不接受二人 checkpoint 或 `--pairs`。
共用安全、统计与输出约定见[公共说明](historical_arena_zh.md)。

## 整队控制与四局轮转

新模型控制一个完整队伍，旧模型控制另一个队伍；对家 0/2 一队，1/3 一队。
每组新旧 Layout 各采两份独立合法布阵，按 `[新 A, 旧 A, 新 B, 旧 B]` 放置，再固定这些布阵轮转四局：

| 轮转编号 | 座位 0（先手） | 座位 1 | 座位 2 | 座位 3 | 新模型整队 |
| --- | --- | --- | --- | --- | --- |
| 0 | 新 A | 旧 A | 新 B | 旧 B | 0/2 |
| 1 | 旧 B | 新 A | 旧 A | 新 B | 1/3 |
| 2 | 新 B | 旧 B | 新 A | 旧 A | 0/2 |
| 3 | 旧 A | 新 B | 旧 B | 新 A | 1/3 |

这里表格是座位编号，不是落子顺序。实际行棋沿用规则引擎逆时针顺序 **0 → 3 → 2 → 1**，淘汰后跳过该座位。
每份布阵都会经过四个物理座位、先手一次；布阵始终随其所属模型，不会把旧模型布阵交给新模型。
四局的行棋随机种子不同，每组重新采样布阵。

一名队友被夺旗或淘汰并不等于整队立即输棋；继续运行到规则引擎给出整队终局。新模型按 `winner_team` 计胜/和/负，同时交叉核对所有座位的最终奖励，已淘汰队友也应获得所属队伍的终局奖励。
**一局只记一个队伍结果；不把同队两个座位重复算两胜。四局平均得分才是一个统计样本。**

## 信息边界

- 四暗：当前 Policy 只知道本座位允许知道的信息，不能读取队友的私有棋子身份或历史。
- 双明：仅按规则引擎显露队友棋子信息，仍不读取对手私有身份。
- 同队两个座位共享该 checkpoint 的权重，但各自有观察历史；共享参数不表示共享私有观察。
- 新旧模型分别使用各自 checkpoint 的历史窗口及合法动作，裁判全知状态不传入 Policy。

## 周期与报表

默认 `--groups 200` 表示 200 个四局组，即 **800 局整队比赛/对手**。
默认每 25 updates 比较固定 baseline + 最近 2 个已评测候选，对手池满时 2400 局/轮，耗时高于同组数的二人评测。
统计结果包含：

- `rotation_groups`：四局组数；`games`：整队比赛局数；`games_per_group=4`。
- `wins/draws/losses`：新模型整队胜/和/负；`score_ci` 按四局组数计算，不能按四倍局数缩窄区间。
- `team_scores`：新模型控制 0/2 队和 1/3 队时的平均得分。
- `rotation_scores`：四个轮转编号的分项平均得分。
- `statistical_unit=four_game_rotation_group`、`result_unit=team_game`、`mode`：用于面板区分二人/四暗/双明。

## 四暗 CPU / 独立设备示例

以下路径为示例，先确认本模式的真实旧 checkpoint 存在。默认 CPU，不启动或恢复训练：

```bash
CUDA_VISIBLE_DEVICES='' python -m junqi.training.evaluate_four_player \
  --mode four_dark \
  --checkpoint-dir runs/four_dark/with_dead_rules/checkpoints \
  --baseline runs/four_dark/with_dead_rules/checkpoints/update_000000100.pt \
  --output-dir eval_history/four_dark_main_v2 \
  --device cpu --cpu-threads 1 \
  --every-updates 25 --recent-opponents 2 --groups 200 \
  --watch --poll-seconds 60
```

双明必须同时替换为 `--mode double_open`、双明 checkpoint 与 baseline 路径，以及独立 `eval_history/double_open_main_v2` 目录。
不能只改 `--mode` 却保留四暗权重或输出目录，程序会拒绝。
作业调度器可用 `--once`；工程短局使用全新输出目录和 `--groups 1 --max-plies 4 --smoke-test --once`，四局短局不代表真实棋力验收。

## Ascend NPU

```bash
EVAL_GAME_MODE=four_dark NPROC_PER_NODE=2 MASTER_PORT=29521 \
EVAL_OUTPUT_DIR=/mnt/shared/eval_history/four_dark_main_v2 \
EVAL_BASELINE=/mnt/shared/runs/four_dark/with_dead_rules/checkpoints/update_000000100.pt \
EVAL_GROUPS=200 EVAL_EVERY_UPDATES=25 EVAL_RECENT_OPPONENTS=2 \
  bash scripts/evaluate_four_player_npu.sh \
  /mnt/shared/runs/four_dark/with_dead_rules/checkpoints/latest.pt --watch
```

四国用 `EVAL_GROUPS`；误设二人的 `EVAL_PAIRS` 或 `EVAL_GAME_MODE=two_player` 会报错。双明设置 `EVAL_GAME_MODE=double_open` 并更换所有模式相关目录。
预先分配独立/空闲 NPU 与独立 rendezvous 端口。多 rank 以完整四局组分片；即使组数不能整除 rank 数，也不会丢局、重复局或拆组计算置信区间。
