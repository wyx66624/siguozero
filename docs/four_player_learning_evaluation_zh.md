# 四国 PPO：训练中怎样判断是否在学习

30 亿步是训练环境交互预算，不是必须跑完才能验收的门槛，也不保证最终棋力。
四暗与双明分别画学习曲线、维护固定对手和评测目录。
应同时回答三个问题：收到有效学习信号了吗、价值预测正常吗、实际下棋变强了吗。

## 每次更新：先看有没有学习信号

训练自动写入各模式的 `metrics.jsonl`、`latest_metrics.json`、`train.log` 和可选 TensorBoard。
新增指标读取本轮已采样的数据，不额外进行模型前向或游戏模拟。
多卡先在一次 collective 中合并原始统计量，再计算全局方差；不平均各 microbatch 的 EV。

| 指标 | 如何解释 |
| --- | --- |
| `rollout/base_games_completed`、`rollout/wins/draws/losses` | 四国 PPO 本轮结束的主局及其胜和负。没有完成对局时不能判定和棋率；长局可能跨多个 update |
| `ppo/raw_nonzero_advantage_fraction` | 标准化前优势非零的样本比例。长期为零时，策略没有从这些优势得到胜负方向的梯度；熵正则仍可能改变权重 |
| `ppo/no_advantage_signal` | 全局本轮优势全部为零时为 1，是排查入口，不是自动判定训练失败 |
| `ppo/raw_advantage_mean/std` | 原始优势的量级与变化；均值为零可能是正负抵消。标准化后的均值、方差不能用来证明有学习信号 |
| `critic/rollout_nonzero_target_fraction`、`critic/rollout_target_std` | 检查价值目标是否全零、恒定或开始出现差异。仅看 `target_mean=0` 不足以判断 |
| `critic/rollout_value_mse` | 采样时冻结 Critic 对本轮 GAE 目标的误差；不是训练完后的独立测试误差 |
| `critic/rollout_explained_variance` | `EV = 1 - Var(target - old_value) / Var(target)`。0 相当于常数预测在方差层面的效果，接近 1 表示解释了较多目标变化，负值提示误差波动较大 |
| `critic/rollout_explained_variance_defined` | 目标方差大于 `1e-12` 才为 1，否则不输出该轮 EV。不能把全零目标、全零预测说成“EV=1，已学会” |
| `policy/approx_kl_old`、`policy/clip_fraction`、`optimizer/*grad_norm` | 检查更新幅度、频繁裁剪、零梯度和数值问题；这些是训练健康指标 |
| `policy/entropy`、`layout/entropy` | 检查是否过早失去探索。熵受合法动作数影响，不能机械地把更低或更高当成更好 |

EV 不敏感于恒定预测偏差，必须结合 MSE 看；目标含 Critic 自身 bootstrap，
所以即使 EV 很高，也不能证明胜负判断准确，更不能代替独立比赛。
指标含义可对照 [Stable-Baselines3 的 PPO 日志说明](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)。

如果连续多个 update 优势全零，先检查期间是否结束了足够多的完整对局、是否全和棋、
终局奖励是否正确进入 GAE，再检查 Critic 输出和训练配置。刚开始尚未走完一局时为零可能正常；
非零优势也可能来自 bootstrap 误差，不能单独当成学会赢棋的证据。
自对弈双方共享同一策略，平均奖励接近零、某一队胜率接近 50% 都不能衡量绝对棋力。

## 周期对弈：画固定对手的得分曲线

现有 `evaluate_four_player` 已支持固定 baseline + 最近历史版本、完整四局轮转、
置信区间、断点续评、冻结快照和中文报告。详见[四国评测](four_player_arena_zh.md)
与[统计协议](historical_arena_zh.md)。评测的是 Policy + Layout 的综合棋力，Critic 不参与行棋。

建议先采用下面的工程观察安排；这些不是已验证的收敛步数：

| 时机 | 内容 | 用途 |
| --- | --- | --- |
| 每个 update（当前约 5120 个全局训练环境交互） | 上述标量、完整对局数和终局分布 | 发现无信号、数值问题及训练异常 |
| 约 10 万～100 万步的早期检查 | 先确认已产生完整对局和非零胜负信号；可做一次 64～128 局对旧模型的完整比赛 | 验证训练链路，较小样本只观察明显问题 |
| 此后每约 100 万步 | 128 局对一个固定 baseline，四局一组轮转 | 观察长期趋势，不因单轮 55% 就宣布提升 |
| 重要阶段，例如预定的 1000 万步检查点 | 对 baseline、较强历史版本及最近版本分别做更多完整对局 | 检查提升、遗忘、针对单一对手过拟合；按所需置信度预先确定样本量 |
| 准备选定一个可用版本时 | 1000 局或更多、未使用的比赛种子，补充人工复盘/人类对战 | 验证相对棋力与实际使用要求；1000 局也不保证能区分很小的差异 |

曲线纵轴用 `score = (wins + 0.5 * draws) / games`，横轴用训练环境交互量或墙钟时间。
固定 baseline 的曲线才能长期对比；每次都换对手的 55% 不能直接连成“棋力提高”的曲线。
四国使用完整四局轮转组作为统计单位，不能把四个座位当四份独立样本。
`score_ci` 下界超过 50% 才是现有协议下领先该对手的证据；跨过 50% 表示证据不足。
训练曲线有噪声、单点成绩有不确定性，参见
[Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)。

达到能稳定击败初始随机模型只是早期里程碑。要判断是否“够用”，需先指定实际目标，
例如对某个固定较强版本的得分、对人类玩家的表现、和棋与超时容忍度。
还应复盘是否存在无意义往返、只会拖和、军旗防守或配合上的明显漏洞。
连续几个预定评测点没有改善时应检查学习信号和训练方案，不能仅凭平台期断言已收敛。

## 现有默认值为什么需要区分

- 训练内的最优模型选择已改为从初始基准起，每 5000 万次训练环境交互选拔一次，
  每轮 100 局，首次在 5000 万步，30 亿步共 60 轮、6000 局。它对阵当时的 best。
  每轮学习信号诊断及自选的早期固定对手评测可更早开展，详见[最优模型选择](best_model_selection_zh.md)。
- 独立历史评测默认每 25 updates 一轮，约 12.8 万步，且默认每个对手 800 局。
  对手池满后是 2400 局/轮；若直接采用，会让早期评测开销很高。
- 周期观察可以另开一个轻量套件：`--every-updates 200 --groups 32 --recent-opponents 0`，
  即约每 102.4 万步对固定 baseline 打 128 局。它与训练内最优模型选择互不替代。
- 自动 `best.pt` 晋级使用点估计得分 >50%；“被晋级”不等于置信区间已经证明更强。
  正式判断应查看完整对弈报告。

频率按全局 `cumulative/environment_plies` 核对。改变全局 rollout batch 时，
`every-updates × 每轮全局实际采样步数` 会改变，不能假设多卡后仍自动对应 100 万步。
独立评测只读取检查点，评测对局消耗资源，但不加入当前定义的 30 亿**训练**环境交互预算。

## 在本机查看与运行

以下是在 WSL 的 `siguozero` 环境、仓库根目录运行的示例。
路径使用 `scripts/start_four_player_ppo.sh` 的默认输出目录，尚未创建的训练目录不会凭空出现。

```bash
conda activate siguozero
cd /mnt/e/agent/siguozero
tensorboard --logdir runs_four_player_ppo_3b --host 127.0.0.1 --port 6006
```

在浏览器打开 `http://localhost:6006`，查看 Scalars 中的 `ppo/`、`critic/`、`policy/`、`rollout/`。
当前 TensorBoard 的横轴 Step 是 **update 次数**，不是环境步数；JSONL 每条记录同时包含
`cumulative/environment_plies`，用于画严格按交互预算的曲线。还可以看 Wall time。
新指标从采用本次代码后的下一轮更新开始写入，旧日志不会自动补算。

以下为每约 100 万步、每轮单个固定对手 128 局的示例。要求该模式已存在早期 baseline；
全新运行且开启模型选择时会创建 `baseline_000000000.pt`，其他情况要替换为真实早期快照。

```bash
python -m junqi.training.evaluate_four_player \
  --mode four_dark \
  --checkpoint-dir runs_four_player_ppo_3b/four_dark/with_dead_rules/checkpoints \
  --baseline runs_four_player_ppo_3b/four_dark/with_dead_rules/model_selection/snapshots/baseline_000000000.pt \
  --output-dir eval_history/four_dark_learning_baseline_v1 \
  --device cuda --cpu-threads 4 \
  --every-updates 200 --recent-opponents 0 --groups 32 \
  --parallel-games 16 --inference-batch-size 16 --environment-workers 4 \
  --once
```

单张 4090 上，CUDA 评测应安排在训练进程安全退出、释放显存后，完成再从 `latest.pt` 续训；
只向进程发送暂停信号不会释放它的显存。不要直接把独立评测的两个模型叠加到满载训练进程。
有独立评测卡时可指定该设备并把 `--once` 改为 `--watch --poll-seconds 60`，自动跟踪新快照。
独立 evaluator 本身不会停止、启动或恢复训练。

报告在 `eval_history/four_dark_learning_baseline_v1/report.md`，配合 `status.json` 判断本轮是否成功。
双明需要同时替换模式、训练目录、baseline 和输出目录。观察到的候选与 baseline update 会写入报告；
原始环境交互计数在对应训练日志中，不要把模型更新次数当作环境步数。

修改同一套件的对局数量、对手池、频率或协议参数时必须使用新输出目录；
不能反复试样本数只保留显著的结果。当前代码不自动启动上述周期评测。

## 评测成本和目前能得出的结论

已有 4090 上 16 局并行完整比赛计时为 130.036 秒，约 8.127 秒/局，
见[完整比赛基准](parallel_arena_benchmark_zh.md)。以相同局长和吞吐线性估算，
128 局约 17 分钟，1000 局约 2.3 小时，模型加载另计。
这只是小样本工程估算；当策略、局长、历史长度变化后需要重新测量。
按此前约 31 个训练交互/秒估算，100 万步训练约 9 小时，
单个固定对手的 128 局评测约增加 3% 时间；默认 2400 局会明显更贵。
可用实测评测时间占比约 5% 作为调度目标，优先调低频率，保留完整规则和对局结算。

已有四暗 10240 步计时实验中完成主局为 0、非零优势比例为 0；双明同规模只记录了 2 场和棋，
非零优势比例也为 0。这些是性能探针，不能证明正式模型已经学会或没有能力学会。
下一阶段的验收应是出现可解释的胜负学习信号，并在独立对手比赛中获得可重复的改进。
不用等到 30 亿步才检查，也不应仅因 loss 下降就继续投入全部预算。
