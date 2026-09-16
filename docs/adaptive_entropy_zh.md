# 行棋策略的自适应探索熵

配置中的 `learner.entropy_coefficient.adaptive_to_target_entropy` 现在控制实际行棋损失和系数更新，适用于四人 PPO 和二人根节点 GRPO。原来的熵奖励只计算一次，更新后的系数用于下一轮训练。

## 损失与反馈

对合法动作分布计算完整联合熵，包含起点选择与该起点下的落点选择：

\[
H(s)=-\sum_{a\in A(s)}\pi(a|s)\log\pi(a|s),\qquad
L=L_{\mathrm{task}}-\mathbb E[\alpha_{\mathrm{phase}(s)}H(s)].
\]

PPO 的任务项保持裁剪策略目标；GRPO 的任务项保留原有策略目标与 reference-KL。行棋反馈使用 `H(s)/log(|A(s)|)`，防止合法走法多的局面天然具有更大熵。只有一个合法动作的局面不参与目标熵反馈；它本身也没有可探索的其他合法动作。

开局前 16 个全局行动（行动前 ply 为 0～15）与后续行动各使用一个独立系数。旧配置共用归一化目标 0.6；新配置可以分别设置目标，并随累计环境步数退火。短历史测试配置会将开局长度限制在可保留的历史容量内，已经滑动的长局不会被重新当成开局。

每个完整 learner update 中保持两个系数不变，所有实际执行的策略优化批次结束后分别更新：

\[
\bar h_t=0.9\bar h_{t-1}+0.1h_t,\qquad
\alpha_{t+1}=\operatorname{clip}\left(
\alpha_t e^{0.1(h_{\mathrm{target,phase}}(n_t)-\bar h_t)},\;
\alpha_{\min},\;\alpha_{\max,\mathrm{phase}}\right).
\]

其中 `n_t` 是本轮开始时已经完成的全局环境步数。首次观测直接初始化滑动平均。平均熵低于目标时增强奖励，高于目标时减弱奖励；某阶段没有可选动作样本时保持该阶段的系数与观测状态。控制器按一次完整训练轮更新，微批数量、梯度累积次数和 OOM 重试不会额外推进它。目标在全部策略 minibatch/epoch 中也保持不变。

`opening_maximum` 可单独设置开局系数上限；未指定时继承 `maximum`。上式的上限因此按阶段分别取值。历史上，2026-09-12 因开局重复、原开局系数已达到 0.02 且目标熵持续未达，本机配置曾将开局上限提高为 0.04，后续阶段仍为 0.02。2026-09-15 四暗配置改为下述较低目标、系数边界与退火计划。

这是有界的工程反馈控制。目标与系数上限均不是已验证最优值，实测熵不会立即等于目标；达到上限后仍可能低于目标。降低系数会按比例减弱相同策略下的熵梯度，但系数上限不等于熵梯度相对任务梯度的比例上限，单凭两项 loss 大小也不能判断梯度谁占主导。后续应同时看开局多样性、完整对局胜负与训练成本。

## 配置

三个现有训练 YAML 均显式包含：

```yaml
learner:
  entropy_coefficient:
    initial: 0.01
    minimum: 0.005
    maximum: 0.02
    adaptive_to_target_entropy: true
    target_ratio: 0.6
    adaptation_rate: 0.1
    ema_decay: 0.9
    opening_plies: 16
```

`opening_target_ratio` 未指定时继承 `target_ratio`。`mode_overrides` 按模式覆盖熵参数，保证本次四暗调整不改动二人和双明训练的目标与边界。三个配置的默认行为保持兼容。

本机 `configs/local_4090_training.yaml` 中只有 `four_dark` 启用：

```yaml
    mode_overrides:
      four_dark:
        minimum: 0.001
        maximum: 0.01
        opening_maximum: 0.02
        target_ratio: 0.50
        opening_target_ratio: 0.50
        target_schedule:
          - [0, 0.50, 0.50]
          - [350000000, 0.50, 0.50]
          - [1500000000, 0.40, 0.40]
          - [3000000000, 0.35, 0.35]
```

每个节点为 `[累计环境步数, 开局目标, 后续目标]`。节点之间分段线性下降，超过最后节点后保持终值；不按训练进程运行时长或 optimizer step 计数，也不随总预算变化重新缩放。当前约 3.25 亿步时目标为 0.50，3.5 亿步后开始连续下降：

| 累计全局环境步数 | 开局目标 | 后续目标 |
| --- | ---: | ---: |
| 现在至 3.5 亿 | 0.50 | 0.50 |
| 15 亿 | 0.40 | 0.40 |
| 30 亿及以后 | 0.35 | 0.35 |

开局系数上限从 0.04 降到 0.02，后续上限从 0.02 降到 0.01，两者下限从 0.005 降到 0.001。两个阶段仍独立反馈；相同目标不代表系数必须相同。调度要求自适应 PPO，目标必须为 `(0, 1]` 的有限数，步数从零开始严格递增，两个阶段的目标分别不递增。没有 `target_schedule` 的配置沿用固定目标。

`opening_plies: 0` 使用单一后续阶段控制器。关闭 `adaptive_to_target_entropy` 时使用固定初始系数。布阵仍使用布阵损失自己的原有熵奖励；这里的行棋目标和阶段统计不套用到布阵指针的合法位置上。

## 恢复、历史模型和统计

- 两个控制器的系数、熵滑动平均与更新次数存入完整 checkpoint 的 `trainer_state.adaptive_entropy`，各模式随各自训练目录恢复。
- 旧 checkpoint 没有此字段时，从当前配置的初始系数开始，无需改模型权重形状或迁移现有进度。显式关闭自适应时采用固定系数；重新开启时不会读取关闭状态下的无效反馈。
- 新配置的上下界会约束恢复后的系数；超出新上限的旧系数在恢复时立即压到上限，保留熵滑动平均和更新次数。每次恢复从检查点的累计环境步数重新计算当前目标，不从退火曲线起点重来。模型权重、优化器与未结束棋局继续按原恢复流程加载。
- 历史模型的非学习回合既不贡献策略梯度，也不贡献探索反馈；所有 rank 都没有学习样本时不推进控制器。
- 阶段统计先累加按样本加权的分子、分母，经过微批和 rank 汇总后再求比值，避免直接平均不同大小的阶段均值。
- 反馈观测来自本轮实际执行的策略学习批次，包括重复 epoch 中的观测；它不是对本轮所有环境状态进行额外一次评测。

## 监控

监控服务的“策略探索”区分别显示开局与后续阶段的归一化熵、当前目标、实际用于本轮的系数、下一轮系数和熵奖励损失贡献；退火启用时还显示最终目标和对应步数。旧日志的目标回退到原共享字段，没有实测值时显示等待记录，不补造零熵数据。

主要日志字段为：

- `policy/adaptive_entropy_enabled`、`policy/entropy_target_ratio`、`policy/entropy_opening_plies`。
- `policy/opening_entropy_target_ratio`、`policy/other_entropy_target_ratio`：本轮两个阶段的目标；兼容字段 `policy/entropy_target_ratio` 表示后续阶段当前目标。
- `policy/entropy_target_schedule_enabled`、`policy/entropy_target_schedule_environment_plies`：是否启用退火及本轮采用的累计步数。
- `policy/entropy_target_schedule_end_environment_plies`、`policy/opening_entropy_target_final`、`policy/other_entropy_target_final`：退火最后节点。
- `policy/opening_entropy_ratio`、`policy/other_entropy_ratio`。
- `policy/opening_entropy_coefficient`、`policy/opening_entropy_coefficient_next`。
- `policy/entropy_coefficient`、`policy/entropy_coefficient_next`（后续阶段）。
- `policy/opening_entropy_coefficient_max`、`policy/entropy_coefficient_max`：两个阶段各自的上限，监控页面随实际日志显示。
- `policy/opening_entropy_coefficient_min`、`policy/entropy_coefficient_min`：对应下限。
- `loss/policy_entropy`：实际加入行棋总损失的负熵奖励。

## 性能与验证

复用行棋训练已计算的动作熵，不新增模型前向，不上传或下载完整动作概率。每个被处理动作增加 8 字节的计数/阶段元数据上传；少量标量统计与已有优化批次指标一起传回。分布式策略指标使用一次批量归约，避免每增加一个指标就增加一次集合通信。

2026-09-12，在 RTX 4090 上用第 935 轮冻结快照、相同 512 个真实开局和后续观测、微批 32、BF16，交替进行 6 组对照。两组系数都为 0.01，额外预热后比较：

| 指标 | 固定系数 | 自适应统计已接入 |
| --- | ---: | ---: |
| 前向＋反向中位耗时 | 0.51759 秒 | 0.52752 秒 |
| 峰值已分配显存 | 422.446 MiB | 422.449 MiB |

该次短批新增耗时约 1.92%，采样起点输出层梯度的最大差值为 0。这里没有执行优化器更新，关闭了 compile，也未测在线采样、Critic、布阵和完整训练轮，因此不能将 1.92% 直接解释为整体训练增幅或用于更新完成时间。

同一批真实观测的开局归一化熵为 0.4472，后续为 0.6478。对应首次反馈将开局系数由 0.01 增至 0.010154，后续系数降至 0.009952，验证了分阶段调节方向。这里开局覆盖前 16 步，与先前仅检查四方首步的诊断范围不同。

另以可控策略验证：胜负优势为零时，熵梯度仍提高低概率合法动作的概率；冻结历史回合不会产生这项梯度。恢复测试覆盖新状态、旧 checkpoint、无样本阶段与多优化批次每轮仅调节一次。完整 Python 回归 354 项，其中 347 通过、7 按环境跳过；页面回归 18 项通过。

上述 2026-09-12 对照的复现工具为 `tools/benchmark_adaptive_entropy.py`，原始数据为 [4090 对照结果](benchmarks/adaptive_entropy_4090_20260912.json)。

2026-09-15 退火测试覆盖分阶段插值、终点保持、非法计划拒绝、四暗模式隔离、真实 PPO 多 minibatch 每轮只反馈一次，以及旧 checkpoint 恢复后保持模型、优化器、RNG、未结束棋局和熵 EMA，只约束旧系数并按已恢复的环境步数选取目标。页面测试覆盖当前与最终目标以及旧日志兼容。目标退火只增加每轮少量 CPU 标量运算，不增加模型前向；这些测试验证实现，不代表棋力已经提升。

本次相关 Python 回归 111 项：108 通过、3 按环境跳过；监控页面回归 19 项通过。正式四暗训练在第 3311 轮、325,484,544 环境步保存完整检查点后恢复；恢复时行棋与布局初始权重逐项相同，原有评测和冠军文件保持一致。第 3312 轮完成 98,304 个新环境步、400 次策略优化，实测两个目标均为 0.50，开局系数为 `0.02 → 0.0197509185`，后续为 `0.01 → 0.0099087713`；EMA 延续及反馈计算均与旧检查点一致。监控 API 已返回退火字段。原始记录位于 `output/diagnostics/entropy_annealing_20260915/verification.json`，首轮指标为同目录 `first_metrics_after.json`。
