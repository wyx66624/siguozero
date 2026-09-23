# MAPPO 战术、棋谱与个人优势日志

当前 expand 训练在每个完成的 update 后写入 `runs_expand_mappo/four_dark/with_dead_rules/tactical_diagnostics/`。
日志只观察真实训练样本，不增加奖励、不调整优势、学习率、采样温度或训练目标。

## 查看入口

- `latest.rank000.md`：最新一轮中文战术表。
- `latest.rank000.json`：最新一轮完整汇总，包括覆盖范围、四位玩家的优势统计、棋局长度和终局状态。
- `updates/update_XXXXXXXXX.rank000.json`：每轮汇总，长期保存。
- `details/update_XXXXXXXXX.rank000.jsonl.gz`：压缩逐步明细，可用 Python `gzip.open(..., 'rt')` 读取。
- `details/update_XXXXXXXXX.rank000.pt.gz`：精确棋谱、初始规则状态和当时的历史输入，供本仓库复盘工具读取。

在 WSL 项目目录运行：

```bash
PYTHONPATH=src python tools/report_mappo_tactics.py \
  --run-dir runs_expand_mappo/four_dark/with_dead_rules --last-updates 20

PYTHONPATH=src python tools/report_mappo_tactics.py \
  --replay runs_expand_mappo/four_dark/with_dead_rules/tactical_diagnostics/details/update_XXXXXXXXX.rank000.pt.gz \
  --verify --game 0 --export-json output/replay_game_0.json
```

将 `XXXXXXXXX` 替换为日志中的轮次。`--verify` 逐步核对合法动作、四位玩家的公开输入和 Critic 输入、
公开战术标注、实际奖励、原始/归一化个人优势以及四个价值目标。复盘文件含 Python 对象，仅读取自己产生的可信本地文件。
不加 `--export-json` 时，会打印所选棋谱的每一步、玩家、动作、优势、价值目标及优化器使用次数。

## 默认范围及成本

每轮轮换抽取 **2 个并行环境槽位**。当前总数 128、每轮 98,304 步，正常情况下保存 **1,536 步/轮（1.5625%）**。
所选槽位内的每一步都保存；一个环境结束后产生的新棋局继续记录。
另从这批动作中按战术类别、选中/未选中分层抽查最多 **96 步**，测量同一输入在策略更新前后的完整合法动作概率。
机会率的分母只是抽样决策，概率探针的分布也不是所有训练步的分布；均不能直接称为全量战术成功率。

明细按照完整 update 成对清理：保留最近 **100 轮**，并受 **1,024 MiB** 上限约束，先达到的限制生效。
极小容量设置可能连当前轮明细也无法保留，汇总中的 `current_details_retained` 会明确记录。
只清理本目录、对应 rank 的日志明细；模型、断点及其他诊断文件不在清理范围。
每轮主训练指标增加 `tactical/*` 和 `timing/tactical_*`，可检查覆盖率、概率探针、序列化和 CPU 标注的成本。
Worker 耗时是多个 CPU worker 的累计时间，不能直接加到墙钟时间中当作减速量。

配置位于 `runtime.tactical_logging`，默认值为：

```yaml
enabled: true
environment_lanes: 2
probe_samples: 96
keep_updates: 100
max_detail_mib: 1024
```

## 战术口径

记录主动炸已知/公开候选可确认的大子、直接吃已知军旗、吃已知小子、工兵挖已知雷、
已知必败碰子、吃兑军旗威胁、工兵/大子避险、工兵暴露于直接威胁，以及无吃子原路返回、
重复走法、后两排连续调动。

所有机会标签只读行动前该玩家可见的身份、公开候选、几何和已有碰子证据。
敌方能否移动和吃兑也在所有公开候选下检查；不调用敌方真实身份生成的合法动作来偷看工兵转弯能力。
真正对局结算中的明棋身份单独放在 `event.referee_outcome`，仅用于事后核查。

`flag_threat_removal` 表示存在合法吃兑动作，能够移除一个公开可确认、当前几何下可以夺旗的敌子。
未选择它不自动判为漏防：玩家也可能堵路、反夺旗或选择其他战术。
`engineer_escape` / `high_piece_escape` 仅指脱离公开可确认的静态一步吃兑威胁；不保证隐藏威胁或多步组合下安全。
原路返回、后排移动或重复走法只记录行为，不自动称为“无意义”。
这些局部模式不替代独立战术测评，也不证明模型棋力提升。

## 一步中保留的信息

每条明细都有游戏 ID、轮次、绝对玩家编号、是否学习席位、旧策略/冻结模型来源、行动者视角下的合法动作、
实际动作、战术机会及未选机会、旧动作概率、个人即时奖励及来源、
`old_value`、`raw_advantage`、`normalized_advantage`、`value_target`，以及四位玩家各自的旧价值和训练目标。
奖励来源同时记录旗子归属、炸弹归属、被炸子力、重复撞子是否触发及本轮实际系数。

这些优势来自**本轮正式训练实际计算出的 GAE**，不是事后换一个 Critic 重算的估计。
Actor 仅使用实际行动玩家自己的归一化优势；另外三个价值目标只属于 Critic。
未选动作没有凭空补上的优势；`missed_chosen_action_raw_advantage_mean` 指有机会却未选时，实际另选动作的优势。
冻结历史席位有棋步与价值标签，但没有 Actor 归一化优势、概率探针和优化器使用计数。

`optimizer_uses` 只在真正的策略优化器 post-step 中计数，包含普通执行与 CUDA Graph 路径；
OOM 重试、空学习批次和 AMP 跳过的 step 不会虚增。
该计数不等于未裁剪的直接梯度次数；正负优势也不保证单个动作概率必然同向变化。
抽样的 `before` / `after` 保存全部合法动作的实际概率，可以检查最终变化。

## 棋谱跨轮边界

训练环境持续运行，抽到时可能已经走了一半。因此每段保存当时的完整规则状态、公开候选/库存、
四个 Actor 历史、四个 Critic 历史及之后的逐步观测。
汇总明确标出 `initial_ply`、`starts_at_initial_position`、`terminal`，从开局到终局的才计为完整棋局。
跨本轮采样边界的未完局保存实际 `bootstrap_values`；不将截断伪装成和棋或终局。
每段可以独立恢复，不依赖前一轮明细还留在磁盘上。后续重新核查时应同时保留本轮目标参数和归一化参数。

## 2026-09-19 正式运行验证

从第 974 轮完整断点（95,748,096 环境步、128 个未完棋局）恢复，服务 PID 526573。
仍采用 `player_decisions` / `player`，没有迁移奖励或重置优化器、随机状态及对局池。
首次完成第 975 轮，记录 1,536 个真实训练棋步、7 段棋谱，其中 3 局从开局到终局完整记录，
另外包含 2 段中途开始并终局的棋谱和 2 段本轮结束时尚未终局的棋谱。
96 个概率探针完成更新前后对照；384 次真实 Actor optimizer step 与正式训练指标一致。

已使用保存文件逐步重放全部 1,536 步，并核对四份公开历史、四份 Critic 历史、合法动作、
公开战术标注、实际奖励、个人原始/归一化优势以及四个价值目标，全部通过。
第 975 轮两份压缩明细合计 3,862,233 字节；父进程记录约 0.49 秒，概率探针约 0.07 秒，
文件写入约 0.64 秒，CPU worker 标注累计约 9.02 秒。worker 累计值不能视作额外墙钟时间。
恢复首轮耗时含预热，不据此宣称长期吞吐或日志开销比例。
随后第 976 轮完成，用时约 100.11 秒；启用前第 971–974 轮为 95.73–112.52 秒。
两侧棋局和 epoch 数不同，这只说明已观察到的后续轮次没有出现大幅耗时增长，不是严格的开销消融。

相关回归 86 项及 71 个子测试通过，追加优化器跳过测试后日志专测 14 项通过。
CPU 开关日志对照的棋局、RNG、目标和最终 Actor 参数完全一致。
恢复前断点元数据和首段可读导出位于 `output/diagnostics/tactical_logging_20260919/`。
