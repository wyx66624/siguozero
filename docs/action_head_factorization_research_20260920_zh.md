# 军棋 PPO：条件分解与完整动作统一 softmax 的文献调研

调研日期：2026-09-20。本文为研究分析，没有修改训练网络、参数或运行中的训练。

**结论。** 两种方式均可用于 PPO。当前的起点→条件终点是自回归分解，不是假定起终点独立。
论文不能支持“平铺一定更适合 PPO”，也不能将当前漏防归因于动作分解。对本项目，完整动作
统一 softmax 是计算成本可接受、值得优先做对照的备选；正式训练方案应由相同预算下的
战术表现和冻结对手评测决定。现阶段建议保留当前训练作为基线，不凭输出形式直接替换。

**比较的对象。** 两边均使用同一个状态下的精确合法动作集合 L(s)。

- 当前：π(i,j|s)=p(i|s)q(j|s,i)，起点和每个起点下的终点分别归一化。
- 备选：每个完整动作有一个分数 z(s,i,j)，π(i,j|s)=exp(z(s,i,j))/Σ_(k,l)∈L(s) exp(z(s,k,l))。
- 静态动作表删除任何合法局面都不可能执行的走法；动态 mask 继续删除当前局面不合法的走法。
  合法但战术上不好的走法仍需学习，不能从 referee 隐藏身份或未来结果构造 Actor 的 mask。
- 将现有联合概率 reshape 成一维，或将现有联合 log probability 再做一次 softmax，不会形成新方案。
  真正需要比较的是动作打分网络及归一化的参数化方式。

**论文证据及适用边界。**

| 论文 | 直接相关发现 | 不应据此推出的结论 |
|---|---|---|
| Huang & Ontañón, A Closer Look at Invalid Action Masking in Policy Gradient Algorithms，2020 预印本 | 证明合法性 mask 与策略梯度兼容；PPO 实验支持在采样和更新中一致使用 masked policy。 | 研究重点是 masking，并未证明平铺优于条件分解。 |
| Kanervisto et al., Action Space Shaping in Deep Reinforcement Learning，CoG 2020 | 多个环境中 Discrete 与 MultiDiscrete 没有显著差异；动作空间扩大时平铺可能较难扩展。SC2 小游戏的 AR 没有显著收益。 | MultiDiscrete 的独立头不等于本项目的条件终点；该文 SC2 实验使用 IMPALA，不能直接当成军棋 PPO 结论。 |
| Silver et al., Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm，2017 AlphaZero 预印本 | 国际象棋使用 4672 个完整走法的联合分布，非法动作置零后重归一化；也测试过平面分布。 | AlphaZero 使用搜索产生训练目标，不是本项目 PPO；动作平面排列也不等于两次条件 softmax。 |
| Vinyals et al., Grandmaster level in StarCraft II using multi-agent reinforcement learning，Nature 2019，及作者架构说明 | AlphaStar 使用自回归动作头和 pointer network，说明这类结构可用于复杂策略游戏。 | 成功系统并不构成其动作头优于军棋平铺输出的受控实验。 |
| Chen et al., Generative Modelling of Stochastic Actions with Arbitrary Constraints in Reinforcement Learning，NeurIPS 2023 | 在特定约束资源分配任务中，AR 表现较差；作者提出前段选择未感知后续约束的解释。 | 约束实验包括 AR+IAR，使用 A2C 体系，与本项目精确逐级 mask 不同；不能把独立分解的表达能力限制套到条件分解上。 |
| Flavin & Sen, Revisiting Action Factorization for Complex Action Spaces，2026-06 预印本 | 比较包含 PPO、SAC、DQN 的 220 个配置；其基准中 Joint 落后，AR 总体有竞争力。 | 尚是预印本，任务、网络容量、参数量及约束机制与军棋不同，不能外推为平铺在所有棋类中更差。 |

原始来源：

1. [Invalid Action Masking](https://arxiv.org/abs/2006.14171)（正文 3.1、4.5）
2. [Action Space Shaping](https://arxiv.org/abs/2004.00980)（正文 III-F、IV）
3. [AlphaZero 预印本](https://arxiv.org/pdf/1712.01815)（Representation，第 12–13 页）
4. [AlphaStar 论文](https://www.nature.com/articles/s41586-019-1724-z)；[作者架构说明](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/)
5. [Generative Modelling of Stochastic Actions](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7d4c0094ae32530494c71468558ab5b1-Abstract.html)（正文 5.2 及实验基线说明）
6. [Revisiting Action Factorization](https://arxiv.org/abs/2606.26574)（正文 7.2.1、9、10；预印本）

**理论上不存在“概率相乘就必然学不好”。** 对任意合法联合分布 r(i,j)，令
p(i)=Σ_j r(i,j)，q(j|i)=r(i,j)/p(i)，即可恢复原联合分布；p(i)=0 的分支不影响结果。
因此，在条件网络容量足够时，分解本身不削弱分布表达能力。有限网络的参数共享方式和
训练过程仍然不同。

当前分解也可写成一个联合 softmax，其等效分数为
z(i,j)=u(i)+v(i,j)−log Σ_(l∈D_i) exp(v(i,l))。
这说明关键差异是起点分数与终点分数的参数化及每行归一化，而不是是否最终存在联合分布。

**与漏防有关的可检验机制，属于本项目分析，并非论文已证实的诊断。**

忽略 PPO clipping、熵和共享骨干的交叉梯度，在单个状态的普通策略梯度下，起点头接收
该起点下的条件平均优势 A_bar(i)=Σ_j q(j|i)A(i,j)。对起点 logit u_k，期望梯度为
p(k)[A_bar(k)−Σ_i p(i)A_bar(i)]。若优势以真实 Vπ 为基准，后一个平均值为零。
一枚棋子的少数防守落点虽好，当前终点策略的大多数走法却很差时，起点头可能被向下更新。
终点头同时会学习调整这些落点的比例，因此不能只看起点头就断言完整防守概率必然下降。
此处平均指同一状态、同一起点的不同落点，不是改变 GAE 或把不同棋局回报混成一个目标。

平铺输出没有这个单独的起点归一化分支，每个完整走法有直接的输出分数；但共享隐藏层、
有限采样、错误优势和 PPO clipping 仍可能造成干扰。平铺不能为从未采到的防守动作自动
生成正优势，也不能修复 Critic 的错误目标。

两种方式的初始探索先验也不同。假设 logits 全相等，起点 A 有 1 个落点，B 有 9 个落点：

| 动作 | 条件分解 | 平铺 |
|---|---:|---:|
| A 的唯一走法 | 50% | 10% |
| B 的每个走法 | 约 5.56% | 10% |

所以分解可能提高低机动性棋子的唯一防守动作概率，也可能降低高机动性棋子的某个关键动作
概率；方向取决于局面。这是等 logits 的数学例子，不是实测当前网络的概率。

**本项目规模与代码核对。** 当前静态动作表含跳过共 5625 项，四人模型上下文宽度 512，
动作头隐藏宽度 256。按带 bias 的线性层计算：

- 当前起点头 512→256→129，加终点头 641→256→129，共 361,986 个参数。
- 若采用 512→256→5625 的平铺头，共 1,576,953 个参数。
- 增加 1,214,967 个参数，相对当前 44,966,018 参数的 Actor 约为 2.70%。

此结果仅为参数量计算；没有运行速度或胜率试验，不能据此声称更快、显存无变化或样本效率更高。
5625 是静态全集规模，每步 softmax 分母只包含动态合法的完整动作。

`src/junqi/training/models.py:2012` 的 `_all_legal_log_probs` 将起点与条件终点的 log probability
相加；`ppo_statistics` 使用完整动作 log probability，并对全部合法动作计算联合熵。
CUDA learner 路径也采用联合概率。当前实现不是两个独立 PPO surrogate，因此没有发现
“单纯因分两层就不符合 PPO”的问题。

**建议的最小对照。** A 保留当前条件分解；B 使用完整动作统一 softmax 和同样的合法 mask。
两组使用相同编码、Transformer/Critic、GAE、奖励、对手协议、环境步预算及多个随机种子。
动作头参数量不同应公开，同时报告等环境步预算和实际耗时。若从当前权重起步，不能直接
用训练好的 A 头与随机 B 头比较；应从相同初始骨干训练，或先把 B 蒸馏到同样的行为分布，
并将蒸馏作为另一个实验条件。

保持采样与更新的 mask 一致，PPO ratio、KL 和熵均按完整动作分布计算。重点记录来自真实
棋谱且按当时公开信息判定的军旗防守、避强子、炸明大子等机会上的联合概率、实际采样量、
优势符号、单轮 PPO 前后的概率变化及完整棋局结果。对照棋力用冻结对手和完整四座轮换组，
报告置信区间；损失下降或训练自博弈胜率不能替代棋力证据。

在完成这一对照前，合理结论是“平铺方案可行且值得测试”，不是“已证明当前动作分解导致漏防”。
