# 四国军棋与二人军棋：仅规则驱动的自对弈强化学习方案

版本：1.4  
设计日期：2026-09-03

## 1. 目标、边界与结论

本方案训练且最终部署两个模型：

1. **布局模型**：棋子条件自回归 Pointer Decoder，按固定棋子序列逐枚从 25 个位置中采样部署点。
2. **策略模型**：Transformer，对当前玩家的信息状态建模，并依次采样起点和终点。

> **实现状态（2026-09-03）**：当前仓库已实现规则引擎、可开关的按观察者确定身份/确定阵亡特征、候选战斗组合唯一性推断、棋子条件 Pointer 布局 Decoder、棋子/位置 Hard Mask、图关系棋盘编码器、512 维时序策略 Transformer、`K=4,M=2` 终局 Game-GRPO learner、单实例全座位共享、原子检查点和三种模式的训练/推理入口。`--no-dead-rules` 会同时关闭推断并从网络中删除阵亡输入分支，两种实验自动保存到不同目录。布局结果仍由 `PlayerSetup` 作最终权威校验。分布式 FSDP、增量 KV cache和历史回归评测调度器属于大规模生产训练的后续性能与调度工作，不影响当前单机闭环的正确性。详细契约见[棋子条件自回归 Pointer 布阵模型与终局训练说明](layout_decoder_training_zh.md)和[死规则特征开关与消融训练](dead_rule_ablation_zh.md)。

两个模型从随机参数开始，四国军棋和二人军棋共享参数，通过模式 token、主视角编码和合法动作掩码区分玩法。训练数据只来自规则引擎中的自对弈，不使用人类棋谱、开局库、专家动作、搜索标签、人工局面价值或中间奖励。

这里的“只喂规则”应工程化地理解为：

```text
规则说明 -> 可执行环境 + 玩家可见观测 + 合法性掩码 + 终局结果
```

不是把自然语言规则作为 prompt 输入模型。每个非终局时刻的游戏奖励严格为 `0`，只有对局结束时产生：

$$
z_i=
\begin{cases}
+1,& i\text{ 所在队最终获胜},\\
0,& \text{和棋},\\
-1,& i\text{ 所在队最终失败}.
\end{cases}
$$

四国军棋中同队两名玩家得到相同的 $z$，另一队得到 $-z$；二人军棋中双方互为相反数。连续 **60 步**没有发生子力交互时，环境立即以和棋结束并返回 `0`。KL 和熵是优化正则项，不是环境奖励，也不向模型注入棋力知识。

主训练算法选择 **GRPO，而不是 DPO**。DPO 面向已有的成对偏好数据和固定参考策略；在线自对弈本身能返回可采样的终局标量结果，GRPO 更直接，也保留行为策略比率和在线探索。DPO 只可作为以后压缩历史检查点的可选实验，不进入本方案主线。

## 2. 研究依据与本方案的创新边界

### 2.1 可直接借鉴的结论

- [DeepNash](https://arxiv.org/abs/2206.15378) 在 Stratego 上从零自对弈，同时学习隐藏布阵和走子，在不进行博弈树搜索、也不使用人类示范的条件下达到人类专家水平。这说明本项目的总体方向可行。它还表明不完全信息策略应保持随机性，并可不设置显式的对手 belief head。
- DeepNash 的空间主干使用 256/320 通道，完整训练使用大规模并行 actor/learner；其成功不能简单归因于某个参数量。本文据此把动作嵌入、棋盘嵌入和布局 token 固定为 256 维，并将动作与棋盘 concat 为 512 维策略时序 token；自对弈吞吐量、策略多样性和长期记忆与模型大小同等重要。
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300) 对同一输入采样一组输出，用组内奖励的均值和标准差构造相对优势，无需另训 critic。本方案将“同一输入”对应到同一个布局起点或同一个对局信息状态。
- [Pointer Networks](https://arxiv.org/abs/1506.03134)、[Neural Combinatorial Optimization](https://arxiv.org/abs/1611.09940) 和 [Attention, Learn to Solve Routing Problems](https://arxiv.org/abs/1803.08475) 支持用自回归分布生成带组合约束的离散解；本方案固定棋子序列，用当前棋子作为 query，并在 25 个位置上施加空位与棋子规则 Hard Mask。
- [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) 表明仅查看最新自身对弈容易漏掉循环退化。本文据此保留布局—策略成对的历史检查点用于离线回归评测和晋级门槛；按本项目的单实例要求，历史检查点不混入在线自对弈座位。
- [PPO](https://arxiv.org/abs/1707.06347) 的裁剪代理目标是本文 GRPO 比率裁剪的基础；[DPO](https://arxiv.org/abs/2305.18290) 的原始目标则是从偏好对直接优化语言模型，并非在线多智能体博弈算法。

### 2.2 不能混同的部分

DeepNash 实际采用卷积 U-Net、价值头、v-trace、NeuRD 和完整的 R-NaD 奖励变换；它不是 Transformer，也不是 GRPO。本文的“阶段参考策略 + KL + GRPO”只是受 R-NaD 启发的工程算法，暂称 **Game-GRPO**，不能宣称继承 R-NaD 的纳什收敛证明。

尤其需要区分：

- R-NaD 的理论陈述针对二人零和博弈，并要求在每个阶段接近正则化博弈的固定点。
- 四国军棋虽然按两个团队给出零和终局回报，但每队由两个拥有不同私有观测、不能自由通信的玩家组成，不能直接套用该证明。
- GRPO 原论文处理同一 prompt 的多条完整输出；本文的“克隆棋局、从同一信息状态分叉到终局”是新的工程适配，需要通过消融实验验证。
- 不设置显式 belief head 不等于忽略历史。策略必须获得当前玩家的完整可见信息、公开推断和足够的历史记忆。

### 2.3 对附件建议的取舍

| 附件中的建议 | 本方案处理 |
|---|---|
| 终局仅给 `+1/0/-1` | 采用；所有中间环境奖励固定为 0 |
| 同一局面分叉并走到终局 | 按最新需求修正为：每个基础决策点从冻结旧策略采样 4 个根动作，每个复制 2 份，共 8 条后缀逐步采样到终局 |
| 不需要显式 Belief Network | 采用；保留完整玩家信息状态、候选集合、按观察者持久保存的确定身份及确定阵亡库存 |
| 起点、终点两阶段动作 | 采用，联合对数概率为两阶段对数概率之和 |
| 熵正则防止策略坍缩 | 采用，并用目标熵自动调节系数 |
| 固定参考策略的 KL | 采用阶段性参考快照，不永久锚定随机初始模型 |
| 称为 Nash-Regularized Game-GRPO | 不作理论命名；只称 R-NaD-inspired Game-GRPO |
| DPO 或 GRPO | 选择 GRPO；DPO 不作为在线自博弈主算法 |

## 3. 环境与信息隔离

### 3.1 两层状态

可执行裁判、主视角观测、公开候选身份、历史事件和动作掩码接口见 [四国军棋与二人军棋自博弈游戏引擎](game_engine_zh.md)。

环境必须同时维护：

- **裁判状态 $s_t$**：包含所有棋子的真实身份、布局、轮次、阵亡信息和随机种子，只供规则结算和分叉模拟使用。
- **玩家信息状态 $I_t^i$**：严格按照玩家 $i$ 当时能观察、记忆和从公开规则推断的内容生成，是策略模型唯一输入。

布局模型只看到己方 25 点部分布阵、当前固定棋子、部署步编号和玩法模式。固定序列已经决定剩余库存，不向布局模型提供对手布局。策略模型不能读取裁判状态，也不能读取由真实暗子身份生成的特权特征或特权合法动作掩码。

建议增加“不可辨识状态一致性测试”：任意交换两个尚未暴露、且公开历史完全等价的敌方棋子身份后，玩家观测、动态动作掩码和模型 logits 必须逐位相同。

### 3.2 主视角

点位和路径直接使用 [玩家主视角棋盘与路径编码规范](board_encoding_zh.md)：

- 四国军棋为 129 个点，自己、左敌、盟友、右敌、中央依次编码。
- 二人军棋为 60 个点，自己始终在 `0..29`。
- 一步策略动作统一为有向二元组 `(from_code, to_code)`。
- 一个训练/推理进程只实例化一个当前 Policy 和一个当前 Layout；四国四个座位或二人两个座位全部复用这两个对象和同一套相对角色 embedding，绝不按座位加载 2/4 份权重。每个座位只拥有独立的主视角历史状态。

### 3.3 公开身份约束

规则引擎可以为审计和规则推断维护每枚暗子的候选身份集合，但棋盘整数链不直接输入 12 位候选掩码。身份尚未确定的非己方棋子显示 `1/2/3`；一旦某位观察者仅凭合法公开信息唯一确定存活棋子的身份，后续棋盘快照就为该观察者持续写入精确的“相对座位 + 棋种”码。实现为每位观察者枚举与公开战果一致的全部进攻/防守棋种组合，只在所有组合一致时提交结论，而不是读取裁判底牌。

确定结论随棋子移动，在棋子阵亡后转入确定阵亡计数，并随中盘检查点保存/恢复。每名其他玩家用与布局生成顺序一致的 25 位二值行；四暗保存上/对/下家共 75 位，双明只保存两名敌方共 50 位并在自己与盟友间共享，二人保存唯一对手 25 位。夺旗或公开出局后该方完整 25 位全为 1。仍有多种解释的同归或吃子不能置位。

死规则包括：司令阵亡亮旗；铁路转弯锁定工兵；工兵存活吃子锁定被吃地雷；吃已知军长且存活锁定司令；已亮旗方吃已知师长且存活锁定军长；已知司令撞死而守子存活锁定地雷；同归时单方新亮旗锁定“司令对炸弹”、双方新亮旗锁定“司令对司令”；已知地雷/军旗同归锁定另一方炸弹；未移动本阵第一排排除炸弹后可锁定同级兑子；确定死亡与确定存活库存耗尽还能排除后续候选。战斗类推断按观察者分别计算，四暗不能把一名玩家的私有结论广播给其他座位。

任何推断只能使用该观察者当时的模式可见身份、既有确定身份、确定阵亡库存、公共候选集合和公开事件，不能用裁判真实身份筛选。推断结果只改变观测棋盘码、确定阵亡位和候选单例，不得产生依赖敌方真实暗子类型的特权合法动作 mask。本次版本化码表覆盖四国 `four_dark/double_open` 和二人 `dark`。

## 4. 256 维动作/棋盘嵌入与 512 维转移 token

布局 Decoder 仍使用 256 维 token。策略模型改为分别生成 256 维动作嵌入和 256 维全棋盘嵌入，再直接 concat 成 512 维时序 token。权威码表和序列边界见 [策略状态、棋盘快照与转移 Token 编码规范](state_token_encoding_zh.md)。

### 4.1 全棋盘整数链

对当前玩家 $i$，按主视角点编码从 0 到末尾排列：

$$
B_t^i=[c_t^i(0),c_t^i(1),\ldots,c_t^i(N-1)],
\qquad N\in\{60,129\}.
$$

核心整数码为：

- `0`：空点；
- `1/2/3`：上家/对家/下家的一枚未知棋子；二人军棋只使用 `2` 表示对手；
- `30..41`：依次按表表示我方军旗、炸弹、工兵、排长、连长、营长、团长、旅长、师长、军长、司令、地雷；
- 已确定身份统一使用 $c(k,r)=b(k)+32r$，其中 $r=0/1/2/3$ 表示我方/上家/对家/下家，对应区间为 `30..41`、`62..73`、`94..105`、`126..137`；
- 工兵四个相对座位码恰为 `32/64/96/128`；双明对家盟友使用对家块 `94..105`；二人已确定的对手也使用对家块；
- 四国四暗模式中，尚未确定的对家盟友棋子仍使用 `2`；padding 为 `138`，词表大小为 `139`。

这里的“整个棋盘信息”是玩家可见且可由规则确定的全棋盘快照，不是裁判全知身份。敌方未确定身份不得由真实棋子类型改变整数码。整数码必须作为类别查表，不得把数值大小直接当军阶强弱。采用步长 32 而不是直接把基础码乘以座位序号，是为了避免“对家司令”和“下家军旗”等类别发生数值碰撞。

### 4.2 棋盘与确定阵亡先验嵌入

策略模型内部的 BoardEncoder 对每个点融合类别码、点位置、点类型、道路/铁路以及模式 embedding，并用图 Transformer 和 `[BOARD]` pooling 得到：

令 $\widetilde D_t\in\{0,1\}^{75}$ 为按相对座位规范化后的阵亡向量。四暗直接使用 75 位；双明在对家盟友槽填零；二人把唯一对手放入对家槽。实现先分别映射，再 concat 融合：

$$
h_t^B=\operatorname{BoardEncoder}(B_t)_{[BOARD]},\quad
h_t^D=W_D\widetilde D_t,
$$

$$
g_t=\operatorname{LN}\!\left(\operatorname{SiLU}\!\left(
W_G[h_t^B;h_t^D]+b_G\right)\right)\in\mathbb R^{256}.
$$

BoardEncoder 同时保留最新棋盘的逐点 256 维表示，供起点/终点 pointer head 使用。经验池保存原始整数链与二值阵亡向量并在 learner 中重新编码，不能保存由旧参数预计算的 $g_t$ 代替原始输入。阵亡融合只增加 151,040 个 Policy 参数，不改变 512 维时序 token 或 1,001 token 上下文长度。

以上是 `dead_rules_enabled=true` 的主实验。关闭时不保存阵亡向量，令 $g_t=h_t^B$，并从模型结构中删除 $W_D/W_G$，而不是给同一网络输入 75 个零；Policy 少 151,040 个参数，Layout 与 token 宽度不变。

### 4.3 动作嵌入与 concat

动作仍以 `(from_code,to_code)` 为核心，加上行动座位和玩家可见的战斗结果，映射为：

$$
q_t=\operatorname{ActionEncoder}(a_t,\text{public result})\in\mathbb R^{256}.
$$

布阵完成后的初始 token 为：

$$
T_0=[\mathbf 0_{256};g_0]\in\mathbb R^{512}.
$$

执行动作 $a_t$ 并完成环境结算后，用动作后的棋盘 $B_t$ 构造：

$$
T_t=[q_t;g_t]\in\mathbb R^{512},\qquad t\ge1.
$$

顺序固定为“动作在前、棋盘在后”，只能 concat，不能相加。初始 token 的动作半区必须是严格全零，而不是学习的特殊 embedding。

### 4.4 1,000 步上下文

初始 token 始终保留，再保留最近最多 1,000 个“动作 + 动作后棋盘”转移 token：

$$
S_t=[T_0,T_{\max(1,t-999)},\ldots,T_t].
$$

因此策略时序最长为 1,001 个 512 维 token。第 1,001 个动作到来后淘汰最老的转移 token，但不淘汰 $T_0$。最后一个 token 的棋盘半区始终表示当前完整可见棋盘；所有仍存活的已确定棋子在最新快照里继续使用精确码，所以不会因原始推断事件滑出窗口而遗忘。

### 4.5 布局位置 token 与棋子 query

布局模型不使用策略模型的 512 维动作—棋盘转移 token。第 $t$ 步先把 25 点部分布阵 $B_{t-1}$ 编成 256 维位置 token：

$$
x_{t,j}=E_{\text{point}(j)}+E_{\text{occupant}(B_{t-1,j})}
+E_{\text{row}(j)}+E_{\text{point-type}(j)}+E_{\text{mode}}.
$$

位置 Transformer 产生逐点表示 $H_t$ 和 pooling 后的 $h_{global}$。当前固定棋子 $c_t$ 作为 Pointer Query：

$$
q_t=\operatorname{MLP}([E_{piece}(c_t);E_{step}(t);E_{mode};h_{global}]).
$$

固定棋子序列与步编号已经唯一确定剩余库存，因此不再输入 12 维剩余数量向量。

## 5. 模型一：棋子条件自回归 Pointer Decoder

本节给出核心概率定义；固定棋子与位置顺序、部分布阵编码、Hard Mask、终局重放字段和验收要求见[棋子条件自回归 Pointer 布阵模型与终局训练说明](layout_decoder_training_zh.md)。

### 5.1 固定棋子序列与自回归分解

固定部署顺序为：

```text
军旗，地雷×3，炸弹×2，司令，军长，师长×2，旅长×2，
团长×2，营长×2，连长×3，排长×3，工兵×3。
```

令 $c_t$ 为第 $t$ 个固定棋子，$p_t$ 为网络采样的位置，$B_{t-1}$ 为已有部分布阵：

$$
P_\phi(p_{1:25}\mid C,m)=\prod_{t=1}^{25}
P_\phi(p_t\mid c_t,B_{t-1},m).
$$

棋子种类和数量由固定序列保证；网络每一步只输出 25 个位置 score，不输出棋子类型。

### 5.2 Piece-conditioned Pointer Head

对当前棋子 query 和 25 个位置表示计算：

$$
s_{t,j}=\frac{(W_Qq_t)^T(W_Kh_{t,j})}{\sqrt{d_k}},
\qquad j=1,\ldots,25.
$$

其中 `Q` 表示当前棋子、步骤、模式和部分布阵全局状态，`K,V` 表示当前部分布阵的 25 个位置。连续部署同类棋子时，步骤 embedding 和已经更新的部分布阵使每次决策保持不同。

### 5.3 Hard Mask 与按概率采样

位置合法性为：

$$
LegalMask_t(j)=EmptyMask_t(j)\land PieceRuleMask(c_t,j).
$$

- 军旗：仅两个大本营；
- 地雷：仅第 5、6 行；
- 炸弹：排除第 1 行；
- 普通棋子：任意剩余非行营点。

非法位置在 softmax 前置为 $-\infty$。由于限制最多的棋子先部署，这个固定顺序可以直接证明不会产生死前缀，不再需要旧位置优先方案的最大流未来可完成性检查。

布局默认随机因子定义为温度 $T_L=0.7$：

$$
P_\phi(p_t=j\mid c_t,B_{t-1},m)
=\operatorname{softmax}(\widetilde s_t/0.7)_j,
$$

$$
p_t\sim\operatorname{Categorical}(P_\phi).
$$

训练、自对弈、历史回归验证和正式部署均按这个最终概率采样，**绝不使用 `argmax`**，也不能在采样后用最高概率位置覆盖结果。

### 5.4 主干和参数量

推荐主配置：

| 项目 | 值 |
|---|---:|
| 架构 | 部分布阵 Transformer + piece-conditioned Pointer Head |
| 层数 | 16 |
| `d_model` | 256 |
| 注意力头 | 8（每头 32 维） |
| SwiGLU `d_ff` | 1,024 |
| 空间位置数 | 25 + 1 个 `[LAYOUT]` pooling token |
| 自回归部署步数 | 25 |
| dropout | 0 |
| 输出 | 25 个位置 Pointer logits |
| 布局温度 | `0.7` |
| 动作选择 | masked categorical sampling，禁止 argmax |
| 估算参数量 | 约 17M，最终以实现统计为准 |

游戏模式使用 `TWO_PLAYER` / `FOUR_PLAYER` embedding。位置 Transformer 对当前 25 点状态可使用双向注意；未来部署尚未写入 $B_{t-1}$，不会造成目标泄漏。

## 6. 模型二：策略 Transformer

### 6.1 主干结构

策略模型由棋盘编码器、动作编码器、512 维因果时序 Transformer 和两阶段动作头组成；它们是一个端到端模型的内部模块：

| 部件 | 冷启动 | 主训练 | 作用 |
|---|---:|---:|---|
| 图 BoardEncoder + 阵亡融合 | 4 层，`d_model=256`、`d_ff=1024` | 8 层，`d_model=256`、`d_ff=1024` | 棋盘池化与 75 位规范阵亡先验分别映射后 concat，融合为 256 维；保留最新逐点表示 |
| ActionEncoder | 256 维 MLP/embedding | 同左 | 编码 `(from,to)`、行动者和公开结果 |
| 因果时序 Transformer | 8 层，`d_model=512`、`d_ff=1024` | 32 层，`d_model=512`、`d_ff=2048` | 处理初始 token 与最近最多 1,000 个转移 token |
| source/destination heads | 256 维 query | 同左 | 与最新棋盘逐点表示做 pointer-style 打分 |

两个 Transformer 都使用 8 个注意力头：BoardEncoder 每头 32 维，时序主干每头 64 维；均使用 Pre-Norm/RMSNorm、SwiGLU 和 dropout 0。模型没有 value/critic head，也没有显式 belief head。

冷启动的 8 层是指 **8 层 512 维时序主干**，不含内部 4 层 BoardEncoder。该配置用于验证棋盘码、concat、终局模拟和 learner；通过启动门槛后把时序主干扩到 32 层、BoardEncoder 扩到 8 层。8 层不是最终冠军容量假设。

时序 Transformer 使用覆盖 1,024 位置的 RoPE 或 ALiBi。初始 token 固定占序列位置 0；最近 1,000 个转移使用位置 1～1000，并附加相对时间/年龄 embedding。推理时缓存 $T_0$ 和滚动窗口 KV；加入第 1,001 个转移前只淘汰最旧转移，不淘汰初始 token。

### 6.2 两阶段动作分布

使用当前仓库的静态动作表和动态合法性掩码。先从可行动起点中采样：

$$
u\sim\pi_\theta^{src}(u\mid I),
$$

再在该起点的合法目标中采样：

$$
v\sim\pi_\theta^{dst}(v\mid I,u),\qquad a=(u,v).
$$

联合概率与对数概率为：

$$
\pi_\theta(a\mid I)=\pi_\theta^{src}(u\mid I)
\pi_\theta^{dst}(v\mid I,u),
$$

$$
\ell_\theta(I,a)=\log\pi_\theta^{src}(u\mid I)
+\log\pi_\theta^{dst}(v\mid I,u).
$$

两个头都推荐用 pointer-style 点积：动作 query 与点 token 的 key 做缩放点积，再施加 `-∞` 合法性 mask。四国/二人由点 mask 和各自的目标槽表自然共享同一个头。

训练和正式对局都从分布采样，默认温度 `1.0`。可以在最终评估阶段研究删除极小概率动作，但必须先验证没有明显增加可利用性，且不能改成全局 argmax。

### 6.3 因子化熵与 KL

策略的精确熵为：

$$
H(\pi_\theta\mid I)=H(\pi_\theta^{src})+
\mathbb E_{u\sim\pi_\theta^{src}}
H(\pi_\theta^{dst}(\cdot\mid I,u)).
$$

对阶段参考策略 $\pi_{ref}$ 的 KL 同样分解：

$$
D_{KL}(\pi_\theta\|\pi_{ref})=
D_{KL}(\pi_\theta^{src}\|\pi_{ref}^{src})+
\mathbb E_{u\sim\pi_\theta^{src}}
D_{KL}(\pi_\theta^{dst}(\cdot\mid I,u)\|\pi_{ref}^{dst}(\cdot\mid I,u)).
$$

合法起点最多 121 个、条件目标槽最多 76 个，因而可在 masked 分布上精确求和，不必用高方差的单样本 KL 估计。计算参考策略时必须使用相同信息状态和相同合法性 mask。

## 7. Game-GRPO：只用终局结果训练策略

### 7.1 每一步从旧策略采样 4 个根动作

生成一盘基础 on-policy 自对弈时，实际落子以及之后的每一步都按冻结旧行为策略 $\pi_{old}$ 的概率分布采样，不使用 argmax：

$$
a_t\sim\pi_{old}(\cdot\mid I_t),\qquad
s_{t+1}=\operatorname{Env}(s_t,a_t).
$$

基础对局中的**每一个决策状态**都成为训练锚点，`anchor_stride=1`。但本版不再穷举合法动作。对锚点 $b$，从 masked 旧策略独立采样 $K=4$ 个根动作候选槽：

$$
a_{b,k}\overset{iid}{\sim}\pi_{old}(\cdot\mid I_b),
\qquad k=1,\ldots,4.
$$

默认是有放回 categorical sampling，重复动作允许出现，因为这正是 $\pi_{old}$ 的采样频率；不得为了凑 4 个不同动作反复拒绝重复样本，否则会改变行为分布。若以后改为无放回采样，必须保存并使用动作的真实 inclusion probability，不能继续套用本节比率。

每个候选根动作把同一裁判状态独立复制 $M=2$ 份，在首步强制执行该根动作。因此每个锚点共有：

$$
G_{trajectory}=K\times M=4\times2=8
$$

条终局续局；这与最新给出的流程图一致。每个副本从下一状态起，所有座位继续调用同一个冻结阶段的旧策略实例随机采样，只是输入换成各自的主视角信息状态：

$$
a_{\tau}\sim\pi_{old}^{\psi_\tau}(\cdot\mid I_\tau^{\psi_\tau}),\qquad
s_{\tau+1}=\operatorname{Env}(s_\tau,a_\tau),
\quad \tau=b+1,\ldots,T-1.
$$

两个副本必须使用独立随机流。基础对局真正执行的动作也继续从 $\pi_{old}$ 采样，不能根据 8 条续局的结果临时选择最高分动作。“所有步骤都模拟”表示基础对局每个决策点都建立上述 8 条续局，且每条一直逐步采样到终局；续局内部不再递归建立新的 8 分支。

### 7.2 两副本平均与 4 候选组内标准化

对第 $k$ 个根动作，先平均两条独立终局结果：

$$
\widehat Q_{b,k}=\frac{z_{b,k,1}+z_{b,k,2}}{2}.
$$

再只在同一锚点的 $K=4$ 个候选槽之间做 group-relative normalization：

$$
\mu_b=\frac1{4}\sum_{k=1}^{4}\widehat Q_{b,k},
$$

$$
\sigma_b=\sqrt{\frac1{4}\sum_{k=1}^{4}
(\widehat Q_{b,k}-\mu_b)^2},
\qquad
A_{b,k}=\frac{\widehat Q_{b,k}-\mu_b}{\sigma_b+\epsilon_A}.
$$

这里同一个根动作若被采样到多个候选槽，各槽保留自己的两次独立续局平均值；在 loss 中自然累加。由于候选已经按 $\pi_{old}$ 采样，组均值本身就是旧策略价值的蒙特卡洛估计，不再对候选额外乘一次 $\pi_{old}$ 权重。

若 4 个两副本均值相同且 `sigma < 1e-4`，该锚点没有相对动作信号，跳过 GRPO 策略项，但仍计算 KL 和熵。终局估值不得加入吃子、子力差、逼近军旗或步数惩罚。

### 7.3 为什么终局只归因给采样的根动作

同一组中的候选只在根部共享 $I_b$，所以 $\widehat Q_{b,k}$ 和 $A_{b,k}$ 只用于对应根动作。根动作之后，各副本进入不同信息状态；不能把根优势复制给整条后缀。基础对局的下一步会单独成为下一个锚点，因此仍满足“每一步都训练”。

该设计是“4 个旧策略根样本 × 每个 2 次终局蒙特卡洛 + root-only GRPO/PPO 裁剪更新”，是针对军棋长程回报的 Game-GRPO 工程变体，不声称具有纳什收敛保证。

### 7.4 采样候选的裁剪损失

对每个候选槽：

$$
\Delta\ell_{b,k}=\ell_\theta(I_b,a_{b,k})-
\ell_{old}(I_b,a_{b,k}),\qquad
\rho_{b,k}=\exp(\operatorname{clip}(\Delta\ell_{b,k},-20,20)).
$$

策略损失为：

$$
L_{P,GRPO}=-\frac{1}{4B}\sum_{b=1}^{B}\sum_{k=1}^{4}
\min\left[\rho_{b,k}A_{b,k},
\operatorname{clip}(\rho_{b,k},1-\epsilon,1+\epsilon)A_{b,k}\right].
$$

这里没有外层 $\pi_{old}$ 权重，因为候选槽本身已经按 $\pi_{old}$ 抽样；再次加权会错误地变成近似 $\pi_{old}^2$。联合 log-prob 仍由起点和条件终点两阶段相加。

旧策略 `old` 是生成当前 4 个根候选和全部后缀动作的逻辑行为版本；它与跨多个更新保持不变的阶段参考策略 `ref` 不是同一个概念。当前同步实现先把唯一的 current Policy/Layout 切到 eval，在完整采样阶段禁止优化器更新，再保存旧两阶段 log-prob，因此无需物理深拷贝一份 behavior，更不得为每个座位深拷贝。每个候选槽必须保存行为模型版本、根动作、旧两阶段 log-prob、两个副本种子和两个终局值，超过一个行为版本的样本直接丢弃。

### 7.5 更新批次边界

一个锚点固定需要 8 条完整终局续局，4 个候选槽及每槽两个副本是不可拆分的逻辑组。默认全局 Policy batch 为 128 个锚点：

$$
B_{anchor}=128,\qquad
N_{continuation}=128\times4\times2=1,024.
$$

显存充足时可扩到 256 个锚点、2,048 条终局续局；冷启动最低不小于 96 个锚点、768 条终局续局。物理 microbatch 可以通过梯度累积拼成全局 batch，但组内均值和标准差必须在完整 4 候选上计算。至少运行 256 个并行环境。

## 8. 用终局结果训练布局模型

本节的关键约束是：布局轨迹只有在对应游戏得到真实终局结果后才可进入 learner。生产训练可以等待一个完整候选组再统一反向传播；最小在线调试可以逐盘更新。二者都不得在游戏中途给布局参数施加胜负梯度。完整数据流见[布局 Decoder 与终局训练说明](layout_decoder_training_zh.md)。

布局没有中间动作回报。对每个模式和对手上下文，从同一个空布局输入采样 $G_L$ 个完整合法布局。每个 group 只替换一个焦点座位的候选布局，其余座位使用跨候选一致的检查点与布局随机种子；四国模式轮换四个焦点座位，不能在同组内同时改变两名盟友的布局而混淆归因。每个候选布局与相同的一组对手检查点、先手和座位排列进行 $M$ 盘终局 rollout：

$$
R_{bk}=\frac{1}{M}\sum_{q=1}^{M}z_{bkq}.
$$

$R_{bk}$ 只是多个合法终局结果的平均，不含中间塑形。随后在同组布局之间标准化得到 $A^L_{bk}$。

对第 $t$ 个固定棋子的 Pointer 位置选择比率：

$$
\rho^L_{bkt}=\exp\left[
\log P_\phi(p_{bkt}\mid c_t,B_{bk,t-1},m)-
\log P_{old}(p_{bkt}\mid c_t,B_{bk,t-1},m)
\right].
$$

布局损失采用 token 级裁剪，整条布局共享终局优势：

$$
L_{L,GRPO}=-\frac{1}{BG_L\cdot25}
\sum_{b,k,t}\min\left[
\rho^L_{bkt}A^L_{bk},
\operatorname{clip}(\rho^L_{bkt},1-\epsilon_L,1+\epsilon_L)A^L_{bk}
\right].
$$

不使用 25 个概率比率的乘积，因为长序列乘积会造成严重的方差和数值不稳定。布局模型和策略模型的检查点始终成对归档和评测，避免用“新布局 + 不匹配的旧策略”错误评价布局。

## 9. 总损失与阶段参考策略

### 9.1 总损失

$$
L_P=L_{P,GRPO}+\beta_P\,\mathbb E[D_{KL}(\pi_\theta\|\pi_{P,ref})]
-\alpha_P\,\mathbb E[H(\pi_\theta)],
$$

$$
L_L=L_{L,GRPO}+\beta_L\,\mathbb E[D_{KL}(p_\phi\|p_{L,ref})]
-\alpha_L\,\mathbb E[H(p_\phi)].
$$

所有 KL 和熵只在合法支持集上计算。若某一步只有一个合法选项，该步熵为 0，不制造伪探索。

### 9.2 参考策略更新

维护三类算法角色，但仍只有两个模型种类：

- `current`：各一份当前布局模型和策略模型；所有玩家共享，也是唯一执行座位动作的模型对。
- `old`：生成一个 rollout 批次时的 current 行为版本及已保存 log-prob；同步实现不是额外 Module。
- `ref`：各一份训练阶段内冻结的布局—策略参考模型，仅用于 KL，不控制任何玩家。

阶段开始时 `ref <- current`，在阶段内固定。候选模型通过历史回归验证后，才执行下一阶段的 `ref <- current`。参考绝不能永久固定为随机初始网络。该过程模拟 R-NaD 的分阶段正则化思想，但一次 GRPO 阶段不等于求得正则化博弈固定点，故不附带纳什保证。

### 9.3 自适应 KL 与熵

使用双变量控制器，而不是永远固定系数：

$$
\beta\leftarrow\operatorname{clip}
\left(\beta\exp[\eta_\beta(KL_{ema}/KL_{target}-1)],\beta_{min},\beta_{max}\right),
$$

$$
\alpha\leftarrow\operatorname{clip}
\left(\alpha\exp[\eta_\alpha(H_{target}-H_{ema})/H_{target}],\alpha_{min},\alpha_{max}\right).
$$

熵监控使用 `H / max(log(number_of_legal_actions), 1)` 的归一化值，使二人、四人以及不同残局可比较。当策略过于确定时增大 $\alpha$；当策略偏离阶段参考过快时增大 $\beta$。

## 10. 单实例共享自进化与历史回归评测

### 10.1 玩家模型硬约束

每个模式的一个在线自对弈进程只允许存在下面这一套当前玩家模型：

```text
shared_current = (1 x layout_model, 1 x policy_model)
```

环境保存四个或两个不同的玩家信息状态，但每次行动都调用 `shared_current.policy`；开局的全部座位布局由 `shared_current.layout` 一次批量采样。代码、配置和日志都断言玩家 Policy/Layout 实例数为 1。同步 actor 与 learner 分阶段运行，rollout 期间 current 参数不更新，所以 `old` 只需保存行为版本和 log-prob，不创建按座位模型，也不创建冗余 behavior 深拷贝。

KL 所需的 `reference = (1 x reference_layout, 1 x reference_policy)` 是只读正则化对象，不参与玩家行动。因而训练常驻的是一套 current 和一套 reference，而不是四个玩家各自一套；纯推理只加载 current 两个模型。

### 10.2 二人军棋

同一个 Policy 依次为双方行动，同一个 Layout 为双方概率采样不同合法阵形。双方输入先分别旋转到自己的主视角，私有历史也分别保存；输入不同不会产生参数副本。先后手和上下方应在不同基础局中轮换。

### 10.3 四国军棋

同一个 Policy 依次控制四个座位，包括两队全部成员；同一个 Layout 批量生成四份独立随机布局。队友和对手都只读各自信息状态，不能交换私有信息。物理座位、先手和左右敌位置在基础局间轮换，并分别统计两队终局回报。四人模式只声称学习共享参数的团队策略，不声称计算了严格纳什均衡。

### 10.4 历史检查点只用于训练外评测

每 10,000 次策略更新或每次参考候选产生时，保存冻结模型对：

```text
checkpoint_n = (layout_model_n, policy_model_n, rule_version, metrics)
```

在线训练局不得把历史 Policy 分配给某些座位。历史检查点用于独立回归评测进程，检查最新模型是否遗忘或出现循环退化，并决定是否刷新 reference；评测进程和训练进程的显存、吞吐及日志必须分开核算。即使专门的交叉版本评测需要同时加载 current 与一个历史对手，也只按独特版本加载，绝不按四个座位无条件复制四份。

### 10.5 三种玩法

正式训练提供四暗、双明和二人军棋三个独立入口；每种模式再按 `with_dead_rules/without_dead_rules` 分离日志、中间结果和检查点，避免不同信息模式或不同网络结构互相覆盖恢复状态。检查点同时校验模式和开关。若以后进行联合实验，模式 token、相对座位 token 和合法 mask 是必要分支；每个训练进程内仍必须满足全座位共享一套 current 模型的约束。

## 11. 推荐模型规模与训练超参数

### 11.1 图示初始参数审核

附件表格整体适合作为冷启动配置，正式值保存在 [configs/bootstrap.yaml](../configs/bootstrap.yaml)。为适配本方案的“仅终局 Game-GRPO”，按下表采纳：

| 参数 | 采纳的启动值 | 审核结论 |
|---|---:|---|
| Action embedding dim | 256 | 采用；每一步动作字段映射为 256 维 |
| Board embedding dim | 256 | 新增；每个完整玩家可见棋盘链映射为 256 维 |
| Transition token dim | 512 | 两个 256 维向量按 `[动作; 动作后棋盘]` concat；初始 token 的动作半区全为 0 |
| Transformer `d_model` | 布局/BoardEncoder 为 256；策略时序为 512 | 当前 token 定义覆盖附件中的单一 256 维值 |
| Transformer layers | 布局 8；BoardEncoder 4；策略时序 8 | 仅作冷启动；三个数字指独立模块，不能再解释为 `3/3/2` |
| Attention heads | 8 | 采用；256 维模块每头 32 维，512 维时序模块每头 64 维 |
| FFN | 1,024 | 冷启动采用；主训练策略时序主干扩为 2,048 |
| history length | 固定初始 token + 最近最多 1,000 个转移 | 后续需求覆盖原值；总时序长度最多 1,001，当前棋盘始终在末 token 中 |
| 并行环境 | 256+ | 作为最小并行数采用，可随硬件水平扩展 |
| rollout / update | 默认 1,024 条终局续局 | 128 个锚点 × 4 个旧策略根样本 × 每个 2 个副本；每条不得截断 |
| learner batch | Policy 为 128 个完整锚点组；布局为 1,024 个序列 | 1,024 不适合作为 1,001-token 策略序列数；Policy 可在 96～256 锚点间按显存调整，实际优化器仍为 AdamW |
| epochs | 最多 3 | 采用；KL 超限或 clip fraction 过高时提前停止 |
| GRPO 根候选 `K` / 副本 `M` | `K=4, M=2` | 共 8 条终局续局；先对同根的 2 个结果平均，再在 4 个候选槽间标准化 |
| clip $\epsilon$ | 0.2 | 启动阶段采用；规模放大后可降到 0.15 |
| entropy coeff | 0.005～0.02 | 采用；从 0.01 开始并按目标熵自适应 |
| 布局随机因子/温度 | `0.7` | Hard Mask 后除以 0.7 做 softmax，再按最终概率分类采样 |
| 布局部署选择 | categorical | 训练、验证和部署均禁止使用 `argmax` |
| learning rate | `1e-4` 起 | 两个模型冷启动均采用；随后由 KL 和阶段晋级动态调节 |
| gradient clipping | 1.0 | 直接采用，指全局梯度范数 |
| terminal reward | `+1/0/-1` | 直接采用；非终局奖励严格为 0 |
| $\gamma$ | 1.0 | 直接采用；终局结果不按对局长度折扣 |

每个锚点固定产生 `4×2=8` 条终局续局。默认收齐 128 个完整锚点组，即 1,024 条已完成模拟后更新；显存充足时可用 256 个锚点和 2,048 条续局。只有走到胜、负或和并取得终局 $z$ 后，该根动作才可进入 learner，不能用固定环境步截断。

3 epochs 不是强制重复三遍。每轮都重新计算 KL、clip fraction 和有效样本数；当 `KL > 1.5 × target_KL` 或被裁剪样本比例超过 30% 时立即结束本批优化。行为数据最多保留一个策略版本。

### 11.2 模型规模阶梯

| 配置 | 布局模型 | 策略模型 | 总参数量（实测统计） | 用途 |
|---|---:|---:|---:|---|
| 冷启动版 | 8,887,296 | 26,610,696 | 35,497,992 | 验证 Pointer、Hard Mask、concat、loss 和吞吐 |
| 主训练版 | 17,292,288 | 144,157,704 | 161,449,992 | 首个顶尖棋力目标 |
| 扩展版 | 25,697,280 | 215,534,600 | 241,231,880 | 主训练版仍受容量限制时 |

所有模块都使用 8 头；布局和 BoardEncoder 的 `d_model=256`，策略因果时序主干的 `d_model=512`。参数量是按标准 attention、SwiGLU、embedding 与动作头做的量级估算，最终以实现后的参数统计为准，而且每个当前模型只计一份，不按座位乘四。不能只因算力充足就直接无限增大网络：自对弈博弈的瓶颈往往是有效终局样本、隐藏信息记忆和策略多样性。只有在相同数据量下，扩展版的训练/验证 loss、历史回归最差分位胜率和布局多样性都改善时才升级。

DeepNash 论文公开的是 256/320 通道网络和极大训练基础设施，没有给出“达到顶尖水平至少需要多少 Transformer 参数”的可迁移定律。因此上述约 162M/243M 是本项目在 512 维时序 token 定义下的工程起点，不冒充论文结论。

### 11.3 规模放大后的优化值

下表用于冷启动通过后、扩到主训练版时；图示的实际冷启动值以 11.1 节和配置文件为准。

| 超参数 | 布局模型 | 策略模型 |
|---|---:|---:|
| 优化器 | AdamW | AdamW |
| `betas` | `(0.9, 0.95)` | `(0.9, 0.95)` |
| `eps` | `1e-8` | `1e-8` |
| weight decay | `0.05` | `0.05` |
| 峰值学习率 | `5e-5` | `1e-4` |
| 学习率下限 | `5e-7` | `1e-6` |
| warmup | 2,000 更新 | 2,000 更新 |
| 梯度范数裁剪 | `1.0` | `1.0` |
| GRPO clip | `0.15` | `0.15` |
| 每批优化 epoch | 1 | 1 |
| 初始 `beta_KL` | `0.02` | `0.02` |
| 目标 KL | `0.010/token` | `0.015/action` |
| 初始 `alpha_entropy` | `0.005` | `0.003` |
| 目标归一化熵 | `0.35` | `0.30` |
| `epsilon_A` | `1e-4` | `1e-4` |

系数范围建议 `beta ∈ [1e-4, 0.2]`、`alpha ∈ [1e-4, 0.05]`，控制器步长 `eta_beta=eta_alpha=0.05`，使用 100 次更新的 EMA。

学习率可变策略：warmup 后每个参考阶段内余弦下降到该阶段峰值的 20%；若观测 KL 超过目标的 2 倍则立即将学习率减半，若 KL 低于目标一半且历史回归分数连续 2,000 更新无提升，可提高 20%，但不超过初始峰值。每次参考晋级后把下一阶段峰值乘 `0.9`，直至学习率下限。

### 11.4 rollout 方差扩展值

策略训练对每个锚点从冻结旧策略采样 4 个根动作候选，每个候选完成 2 次独立随机后缀。冷启动和主训练都使用这一规则：

| 项目 | 二人 | 四国 |
|---|---:|---:|
| 基础对局锚点间隔 | 1（每一步） | 1（每一步） |
| 旧策略根候选槽 `K` | 4 | 4 |
| 每候选独立副本 `M` | 2 | 2 |
| 每锚点终局续局 | 8 | 8 |
| 默认全局锚点 batch | 每个独立模式 128 | 每个独立模式 128 |
| 默认每次更新终局续局 | 1,024 | 1,024 |
| 最少并行环境 | 256 | 256 |
| 布局候选数 `G_L` | 冷启动 8，主训练 32 | 冷启动 8，主训练 32 |
| 每候选布局终局数 `M` | 4 | 8 |
| 布局批次上下文数 `B_L` | 64 | 64 |
| 策略更新 : 布局更新 | 8 : 1 | 8 : 1 |
| 策略时序上下文 | 固定 $T_0$ + 最近 1,000 个转移 | 固定 $T_0$ + 最近 1,000 个转移 |

若一个状态的 4 个两副本均值完全相同，跳过该状态的策略项，并通过更多基础对局获得不同锚点，不添加奖励塑形。生产级目标必须分别记录基础对局、锚点、终局续局、分叉环境步和 learner update。随机规则引擎基准、显存、batch 口径及总预算的推导见[训练计算量、显存、批大小与总样本预算](compute_budget_zh.md)。

### 11.5 计算资源结论

当前随机策略基准中，二人平均约 327 步，四国平均约 611 步。改为每锚点固定 8 条续局后，一盘基础二人/四国对局约派生 2.6K/4.9K 条终局续局，相比旧穷举方案分别下降约 15.1/21.2 倍。

正式启动值为：Policy 全局 batch `128` 个完整锚点组；主模型单卡 microbatch 在 RTX 4090 24GB 上用实测安全值 `8`，H100 80GB 从 `32` 起测，H200 141GB 从 `64` 起测，再以梯度累积补足。布局模型仍使用全局 batch `1,024`。冷启动以约 `1M` 个锚点、`8M` 条终局续局验收，主训练至少规划 `8M` 个锚点、`64M` 条续局；冲击顶尖人类按 `25.6M～76.8M` 个锚点、`0.205B～0.614B` 条续局、约 `200K～600K` learner updates 预留。该范围是容量规划，不是棋力保证；完整实测表见[三模式训练设置、资源需求与运行手册](training_resources_zh.md)。

## 12. 训练阶段

所有阶段均使用完整原规则、相同终局奖励和“全座位共享一套 current 模型”约束；“阶段”只改变哪些参数冻结及参考刷新节奏，不简化胜负条件。

### 当前执行顺序：先训练二人军棋

自 2026-09-03 起，首个长程训练任务只运行 `two_player`，四暗和双明在二人模式通过规则、吞吐、断点恢复和历史回归门槛后再启动。首个二人阶段把 **30 亿步**严格定义为分叉续局中的累计环境行动数，即日志字段 `cumulative/continuation_plies`；它不是 optimizer step、learner update、基础局步数或 token 数。

按二人随机基准每条终局续局平均剩余 `182.5` 步、默认 `128` 个锚点和每锚点 `4 x 2 = 8` 条续局估算：

$$
N_{rollout}\approx\frac{3.0\times10^9}{182.5}=16.44\ \mathrm{M},\qquad
N_{anchor}\approx\frac{N_{rollout}}{8}=2.055\ \mathrm{M},
$$

$$
N_{outer\ update}\approx\frac{N_{anchor}}{128}=16{,}054.
$$

每个外层 update 的 Policy 数据最多复用 3 个 epoch，因此约对应 `16,054～48,162` 次 Policy `optimizer.step()`。这些换算只用于规划；实际停止条件应在完成某个原子 update 后检查累计 `continuation_plies >= 3,000,000,000`，不能根据预估 update 数假装精确命中。

30 亿步约为二人冷启动预算 `1.46B` 分叉步的 2.05 倍、正式主训练预算 `11.68B` 的 25.7%。因此它适合作为“冷启动后扩大验证”的首个里程碑，但不能作为战胜顶尖人类的最终样本保证。分别在 `0.5B / 1.0B / 1.5B / 2.0B / 3.0B` 步冻结候选并做训练外评测；若连续三个评测点没有统计显著进步，或出现非法动作、信息泄漏、熵坍缩和异常拖和，应提前停止或回滚。

2026-09-03 在当前单张 RTX 4090 24GB、WSL2、PyTorch 2.11.0+cu130 上进行的二人 bootstrap 端到端探针完成 `64` 条终局续局、`22,011` 个分叉步，用时 `158.51 s`，实测 `138.86` 分叉步/秒。照此速度，30 亿步需要约 `250` 个连续运行日；按 90% 可用率为约 `278` 天，考虑长历史、评测和中断应按 `9～12` 个月规划。该结果表明当前瓶颈是 Python 规则 Actor 和重复全历史编码，不是 4090 显存。长跑启动门槛设为端到端持续吞吐至少 `2,000` 分叉步/秒；达到该门槛后，30 亿步约需 `17.4` 个连续运行日或约 `19.3` 天（90% 可用率）。在达到吞吐门槛前，只运行短程正确性、性能剖析和优化实验，不静默启动数月任务。

### 阶段 A：规则与随机策略验收

- 布局 Pointer logits 全零，按固定棋子序列在 `EmptyMask AND PieceRuleMask` 后以 `T=0.7` 均匀采样合法位置；即使均匀也必须走 categorical 路径，不能调用 `argmax`。
- 策略 logits 全零，在动态 mask 后均匀采样合法动作。
- 对每种受支持的信息模式检查棋盘码白名单、初始动作半区全零、动作后快照对齐、`512` 维 concat，以及超过 1,000 步时只淘汰最老转移而保留 $T_0$。
- 跑至少一百万次布局和十万盘随机对局，确保无非法动作、无信息泄漏、60 步无交互严格判和。

### 阶段 B：先稳定策略模型

- 使用 [configs/bootstrap.yaml](../configs/bootstrap.yaml) 中经审核的 8 层启动模型和超参数。
- 暂时冻结随机布局分布，只优化策略模型。
- 当前先只训练二人军棋，达到上述 30 亿步阶段目标并通过独立评测后，再启动四暗和双明；不同模式不混用检查点。
- 根分叉 GRPO 能稳定区分胜/和/负后进入下一阶段。

### 阶段 C：加入布局模型

- 固定策略 8 个更新块后，冻结策略并更新布局 1 个块，交替进行，避免双方同时快速漂移。
- 从本阶段开始，每盘结束后都记录各座位的完整布局轨迹与其终局 `+1/0/-1`；只有具备焦点座位、候选组和共同随机数元数据的完整配对原子组才进入布局反向传播。8:1 表示优化器步频率，不表示布局模型脱离自对弈。
- 每个布局候选用成对共同随机数和座位轮换降低方差。
- 检查布局熵、军旗位置比例和完整阵形频率，防止单一开局坍缩。

### 阶段 D：历史回归评测和阶段参考

- 在线自对弈仍由唯一 current 模型控制所有座位，不向座位注入历史模型。
- 每 50,000 次策略更新形成晋级候选，在独立评测任务中对当前 reference、最近 10 个检查点和循环克制检查点做座位均衡测试；未通过则保持原 `ref`，降低学习率或增大 KL，而不是无条件覆盖。
- 晋级后归档当前布局—策略对，并开始下一参考阶段；历史集合只服务回归评测。

### 阶段 E：规模扩展与人类评测

- 在主训练版性能仍随数据稳定上升时继续扩展 actor 数量；只有容量诊断成立时改用 48 层策略模型。
- 人类棋谱只用于最终评测记录，不进入梯度、经验池或超参数逐局调优。
- 正式评测仍按概率采样策略，不公布或复用评测对手的私有布局。

## 13. 参考晋级与“战胜顶尖人类”的验收

### 13.1 自动晋级门槛

每个候选在训练外评测任务中与以下集合进行座位均衡的独立比赛：当前参考、最近 10 个检查点、循环克制最强的 10 个检查点、随机合法基线。建议二人至少 20,000 盘、四国至少 40,000 盘。交叉版本评测不改变在线自对弈“全部座位共享 current”的硬约束。

同时满足才晋级：

1. 相对当前参考的综合得分提升，95% bootstrap 置信区间下界高于 0；
2. 对历史检查点的最差 10% 分位得分不下降超过 1 个百分点；
3. 两种玩法都不显著退步；
4. 归一化策略熵和布局熵不低于目标下限；
5. 非法动作率严格为 0，信息泄漏测试全部通过；
6. 60 步无交互和棋率没有因无意义循环异常飙升。

### 13.2 顶尖棋力门槛

“自我 Elo 上升”不能证明超过人类。最终需要盲测：

- 预先冻结单一候选，不在测评期间更新；
- 邀请有可验证等级/赛事成绩的顶尖二人和四国玩家；
- 随机座位、先手、队友和多盘布局，评测者不知道模型布局；
- 每种玩法至少 400 盘有效对局；
- 以胜 1、和 0.5、负 0 计分，目标为对顶尖组得分率的 95% 置信区间下界超过 50%；
- 同时报告胜/和/负、座位分解、平均步数和 60 步和棋比例，不能只报挑选后的胜局。

## 14. 关键消融实验

按优先级执行：

1. `root-only GRPO` 对比“整条续局共享优势”；
2. 阶段 KL 对比永久初始参考、无参考；
3. 自适应熵对比固定熵、无熵；
4. 不同 reference 刷新周期和历史回归门槛；在线自对弈座位始终共享 current，不把历史对手混入训练；
5. 固定初始 token + 最近 1,000 个转移，对比固定初始 token + 最近 128/256/512 个转移，并把无上限完整历史作为仅供研究的消融基线；
6. 布局温度 `0.7` 对比 `0.5/1.0`，并分别检查棋力、归一化熵和完整布局去重率；
7. 共享四国/二人主干对比两个独立策略模型；
8. 主方案 `K=4,M=2` 对比 `K=8,M=2`、`K=4,M=1` 和小规模全合法动作基准，区分根动作覆盖与重复估值收益；
9. 32 层对比 48 层策略模型，区分容量瓶颈和数据瓶颈。

每个消融必须使用相同环境决策数和相同的离线评测集合，不以墙钟时间不等的结果下结论。

## 15. 失败模式与防线

| 风险 | 症状 | 防线 |
|---|---|---|
| 暗子信息泄漏 | 换掉不可见身份后 logits 改变 | 两层状态、观测等价测试、禁用全知 mask |
| 布局死前缀 | 某个固定棋子没有合法空位 | 军旗→地雷→炸弹→普通棋子的约束优先顺序、Empty/Rule Hard Mask 和完备性测试 |
| 概率坍缩 | 单一布局/单一动作接近 100% | 目标熵、自适应 `alpha`、历史回归评测 |
| 随机小概率昏招 | 长局必然偶发极差动作 | 后期小概率阈值实验；保持混合策略并做可利用性评估 |
| 4 个候选回报相同 | `sigma=0`、无策略梯度 | 跳过该状态并采集更多不同锚点，不加塑形 |
| 1,000 步以前的行为遗忘 | 无法利用很早的对手模式 | 每步完整棋盘保证当前局面不丢失；用完整历史作消融，若确有收益再显式扩窗或新增版本化摘要 token |
| 续局优势错配 | 不同后续状态共享同一基线 | 主方案只更新共同根动作 |
| 旧数据过期 | ratio 极端、clip 比例过高 | 数据最多保留一个行为版本，存原始 log-prob |
| 两模型共同漂移 | 新布局和新策略互相掩盖退步 | 8:1 交替冻结、成对检查点、参考晋级 |
| 循环克制 | 最新模型胜前任但负更早版本 | 独立历史矩阵 + 最差分位晋级门槛，不改变在线共享模型 |
| 四国座位偏置 | 某些物理座位显著退化 | 同一模型共享参数、主视角旋转、先手与座位轮换 |
| 无意义拖和 | 60 步和棋率持续上升 | 将计数器作为观测；终局和棋仍为 0，不做人为惩罚 |

## 16. 训练主循环伪代码

```python
layout = PieceConditionedLayoutPointerDecoder256.random_init()
policy = GamePolicyTransitionTransformer512.random_init()
reference = deepcopy_pair_once_for_kl(layout, policy)
historical_checkpoints = []

while not champion_gate_reached():
    # Actor/Learner 同步分阶段：采样期间唯一 current 实例不会更新。
    policy.eval()
    layout.eval()
    shared_actor = FrozenPolicyActor(policy)  # 包装器不复制权重
    behavior_version = current_update
    assert shared_actor.policy is policy

    # 基础对局本身也在每一步从旧策略分布采样，不使用 argmax。
    base_games = generate_on_policy_games(
        shared_policy=shared_actor,
        shared_layout=layout,
        player_model_instances=1,  # 四/二个座位全部复用
        modes=("two_player", "four_player"),
        action_selection="categorical",
        history_context={"pinned_initial": True, "recent_transitions": 1000},
    )
    anchor_queue.extend(every_decision_state(base_games))  # stride = 1，不丢步骤

    policy_groups = []
    while len(policy_groups) < 128:
        hidden_state, information_state, root_player = anchor_queue.popleft()
        legal_actions = environment.legal_actions(hidden_state, root_player)
        candidate_actions = shared_actor.sample_actions(
            information_state,
            legal_actions,
            count=4,
            replacement=True,  # iid categorical；允许重复动作
        )
        candidate_old_log_probs = shared_actor.log_probs(
            information_state, candidate_actions
        )

        q_by_slot = []
        replica_rewards_by_slot = []
        for slot, root_action in enumerate(candidate_actions):
            outcomes = []
            for replica in range(2):
                env = clone_environment(
                    hidden_state,
                    replica_seed=(slot, replica),
                )
                env.step(root_action)
                while not env.done:
                    actor = env.current_player
                    obs = env.information_state(
                        actor,
                        history_context={"pinned_initial": True, "recent_transitions": 1000},
                    )
                    action = shared_actor.sample(obs, temperature=1.0)
                    env.step(action)  # sample -> environment，直到终局
                outcomes.append(env.terminal_reward(root_player))
            replica_rewards_by_slot.append(outcomes)
            q_by_slot.append(mean(outcomes))

        advantages = group_standardize(q_by_slot, epsilon=1e-4)
        policy_groups.append(
            {
                "information_state": information_state,
                "legal_actions": legal_actions,
                "candidate_actions": candidate_actions,
                "candidate_old_log_probs": candidate_old_log_probs,
                "replica_rewards_by_slot": replica_rewards_by_slot,
                "advantages": advantages,
            }
        )

    update_policy_once(
        loss=sampled_root_clipped_grpo(policy_groups)
        + adaptive_kl(policy, reference.policy)
        - adaptive_entropy(policy)
    )

    # 布局以较慢时间尺度更新，评价只来自完整对局终局。
    if strategy_updates % 8 == 0:
        freeze(policy)
        layout_groups = sample_legal_layout_groups(
            layout,
            piece_order="flag_mines_bombs_then_regular",
            pointer_temperature=0.7,
            action_selection="categorical",  # 禁止 argmax
            hard_mask="empty_and_piece_rule",
            store_old_log_probs=True,
        )
        outcomes = paired_terminal_self_play(
            layout_groups,
            shared_policy=policy,
            player_model_instances=1,
        )
        update_layout_once(
            loss=clipped_token_grpo(outcomes)
            + adaptive_kl(layout, reference.layout)
            - adaptive_entropy(layout)
        )
        unfreeze(policy)

    if ready_for_validation():
        candidate = immutable_checkpoint(layout, policy)
        if passes_offline_history_gate(
            candidate, reference, historical_checkpoints
        ):
            reference.load_state_dict(candidate)
            historical_checkpoints.append(candidate)
            decay_phase_learning_rates()
```

## 17. 实施顺序与必须记录的数据

实现顺序：

1. 完整规则状态机、终局和 60 步无交互计数器；
2. 裁判状态到四个/两个玩家信息状态的隔离层；
3. 固定棋子序列、25 点 Pointer 编号、棋子条件 Hard Mask 采样器和百万次 fuzz test；
4. 棋盘整数链、BoardEncoder、512 维转移 token 与 1,001 token 滑窗测试；
5. 两阶段策略动作头及 log-prob 回放一致性测试；
6. 终局 rollout、根分叉和 GRPO learner；
7. 成对检查点、训练外历史评测矩阵、参考晋级；
8. 分布式 actor/learner 和完整可复现实验配置；
9. 消融、强基线、盲测人类评估。

每盘至少记录：

```text
rule_version, board_encoding_version, action_encoding_version,
state_token_encoding_version, game_mode, information_mode, rng_seed,
shared_layout_model_version, shared_policy_model_version,
player_policy_instance_count=1, player_layout_instance_count=1,
board_piece_code_version=0.3, exact_identity_stride=32,
per_viewer_known_identities, per_viewer_known_casualty_counts,
per_token_known_casualty_bits, casualty_slot_order=layout_piece_sequence,
layout_piece_sequence_version, layout_point_order_version,
layout_temperature=0.7, layout_position_indices[seat],
layout_legal_position_masks, layout_old_log_probs,
physical_layouts, per-seat observations, full public event log,
sampled actions, player_visible_board_integer_chains,
legal masks, source/destination log-probs,
pin_initial_token=true, recent_transition_limit=1000,
transition_token_dim=512, root_candidate_count=4,
root_candidate_sampling=iid_old_policy_with_replacement,
allow_duplicate_root_candidates=true, terminal_replicas_per_candidate=2,
candidate_root_actions, candidate_old_log_probs,
replica_rng_seeds, replica_terminal_rewards,
eight_terminal_outcomes_per_anchor,
no_interaction_counter, terminal_reason, terminal_z[seat]
```

其中 `physical_layouts` 只能进入裁判日志和离线规则复验，绝不能进入策略模型 batch。训练配置、代码提交、参考版本和单实例共享断言必须随检查点固化，否则自对弈结果不可复现。

## 18. 最终建议

首先采用经新 token 定义修正后的约 36M 参数冷启动版，运行到规则、信息隔离、棋盘码、512 维 concat、`4×2` 根动作终局采样和训练吞吐全部验收，并以约 `1M` 个锚点、`8M` 条终局续局确认 loss 能稳定下降。随后扩到约 162M 参数主训练版，至少运行到 `8M` 个锚点、`64M` 条终局续局；若历史回归评测和盲测仍持续改善，则按 `25.6M～76.8M` 个锚点、`0.205B～0.614B` 条终局续局、约 `200K～600K` learner updates 规划顶尖棋力训练。主训练版继续采用 Game-GRPO、共享且在采样阶段冻结的 current 策略、概率采样 4 个根候选、每候选 2 条独立终局续局、阶段 KL、自适应熵和训练外历史回归评测，并处理每一步的完整玩家可见 129 点棋盘、固定初始棋盘 token，以及最近最多 1,000 个“动作 + 动作后棋盘”转移 token；比盲目堆叠参数更应优先保证：

1. 规则和信息隔离绝对正确；
2. 每一步都从合法概率分布采样；
3. 终局样本量足够大；
4. 布局与策略按成对检查点稳定共同进化；
5. 使用历史回归门槛和熵约束保留稳健的混合策略，同时不复制座位模型。

若主训练版在固定终局样本量下明显欠拟合，再升到约 243M 参数扩展版。若训练胜率只在最新自博弈中上涨、对历史检查点的回归评测或隐藏信息等价测试失败，增加参数不会解决问题，应先修复训练分布或实现错误。

## 19. 参考文献

1. Perolat et al., [Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning](https://arxiv.org/abs/2206.15378), Science, 2022。直接依据：从零自对弈、隐藏布阵、混合策略、R-NaD、网络和训练规模。
2. Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024。直接依据：GRPO 的组内相对优势和 critic-free 裁剪目标。
3. Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017。直接依据：重要性比率裁剪代理目标。
4. Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), NeurIPS 2023。用于说明 DPO 的偏好数据假设及其与在线自博弈的差异。
5. Vinyals, Fortunato and Jaitly, [Pointer Networks](https://arxiv.org/abs/1506.03134), NeurIPS 2015。直接依据：自回归离散组合输出和 pointer-style 选择。
6. Bello et al., [Neural Combinatorial Optimization with Reinforcement Learning](https://arxiv.org/abs/1611.09940), 2016。直接依据：用策略梯度训练组合序列生成器。
7. Kool, van Hoof and Welling, [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475), ICLR 2019。直接依据：attention decoder、合法 mask 与随机/贪心 rollout 比较。
8. Vinyals et al., [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://www.nature.com/articles/s41586-019-1724-z), Nature, 2019。间接依据：保存历史策略并检测循环退化；本方案仅把它用于训练外回归评测。
9. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017。Transformer 主干的基础架构来源。
