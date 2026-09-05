# 棋子条件自回归 Pointer 布阵模型与终局训练说明

版本：0.3  
设计日期：2026-09-02

本文定义布局模型的新权威方案：**Piece-conditioned Autoregressive Pointer Decoder**。模型不再“按照固定位置预测棋子类型”，而是“按照固定棋子序列，为当前棋子预测一个部署位置”。每一步基于已经完成的部分布阵重新计算 25 个位置的概率，在 Hard Mask 后使用默认温度 `T=0.7` 做分类采样，任何阶段都不能用 `argmax` 代替。

总体终局训练算法、走子策略和联赛机制见[仅规则驱动的自对弈强化学习方案](reinforcement_learning_plan_zh.md)。

## 1. 当前代码状态与设计结论

当前可执行代码已经形成从概率布局到终局反向传播的闭环：

| 组件 | 当前能力 | 在训练中的作用 |
|---|---|---|
| `PlayerSetup.random()` | 随机生成合法布局 | 仅作为规则基线和测试夹具 |
| `LayoutBuilder` | 固定**位置**，逐点选择棋子类型 | 保留的旧方向规则接口，不参与神经 Pointer 解码 |
| `LayoutPolicyTransformer` | 固定**棋子**，从 25 个点中概率采样位置 | 新布局策略；保存逐步合法 mask 和行为 log-prob |
| `layout_grpo_loss()` | 使用完整布局轨迹及真实终局结果 | 对布局 Pointer Decoder 做裁剪策略更新 |
| `PlayerSetup` | 校验完整布局的库存和静态规则 | 最终权威校验器 |
| `JunqiGame` | 使用传入的完整布局运行到终局 | 提供唯一胜/和/负回报 |

棋子条件部署状态由 `LayoutPolicyTransformer.sample()` 内部推进：严格按本文固定序列输入当前棋子，生成 25 位 Hard Mask，概率采样后更新部分布阵。轨迹可序列化进检查点，并能由 `layout_sample_from_trace()` 恢复，避免保存不可序列化的规则对象。

新旧方向必须明确区分：

```text
旧设计：固定位置 p_t → 网络从 12 类中选择棋子 c_t
新设计：固定棋子 c_t → 网络从 25 个点中选择位置 p_t
```

本文和强化学习总方案中的新版描述覆盖旧设计；训练源代码已经采用新方向，基础 `LayoutBuilder` 仅为兼容规则 API 而保留。

## 2. 固定棋子部署序列

### 2.1 权威顺序

先部署规则限制最多的棋子，再部署限制少的普通棋子。固定 25 步序列为：

```text
1  军旗
2  地雷    3  地雷    4  地雷
5  炸弹    6  炸弹
7  司令    8  军长
9  师长   10  师长
11 旅长   12  旅长
13 团长   14  团长
15 营长   16  营长
17 连长   18  连长   19 连长
20 排长   21  排长   22 排长
23 工兵   24  工兵   25 工兵
```

记为常量：

$$
C=(c_1,c_2,\ldots,c_{25}).
$$

该序列必须写入版本化常量 `DEPLOYMENT_PIECE_SEQUENCE`，不能直接复用当前按枚举定义的 `PIECE_TYPE_ORDER`。二人和四国模式使用同一个棋子序列。

### 2.2 数量约束由生成空间保证

军旗只在序列中出现 1 次、地雷 3 次、炸弹 2 次，其他棋子也严格按标准库存重复。因此网络只决定“放在哪里”，没有机会额外生成第二枚军旗或第四枚地雷。

数量约束不能放进奖励或 loss 中惩罚，也不需要让模型学习“还剩几枚”：

```text
固定序列决定每种棋子的准确数量
Hard Mask 决定当前棋子能放在哪些空位
模型只学习合法位置之间的策略偏好
```

### 2.3 为什么受限棋子必须在前面

规则限制为：

- 军旗只能放在两个大本营之一；
- 地雷只能放在第 5、6 行；
- 炸弹不能放在第 1 行；
- 司令到工兵的普通棋子可以放在任意剩余非行营布阵点。

按“军旗 → 地雷 → 炸弹 → 普通棋子”部署时，局部 Hard Mask 足以保证后续一定可完成：

1. 军旗开始时有 2 个大本营候选；
2. 第 5、6 行共有 10 个布阵点，军旗至多占用其中 1 个，仍至少有 9 个位置供 3 枚地雷选择；
3. 非第一行共有 20 个布阵点，放完军旗和地雷后仍至少有 16 个位置供 2 枚炸弹选择；
4. 最后 19 枚普通棋子没有额外位置限制，依次填满所有空位。

因此，新固定序列不需要旧位置优先方案中的最大流/二分容量“未来可完成性 mask”。如果以后增加新的棋子部署限制，必须重新证明此性质；证明不成立时再恢复通用可完成性检查。

## 3. 25 个 Pointer 位置的稳定编号

Pointer Head 始终输出 25 个分数，分别对应己方阵地的 25 个非行营布阵点。需要新增独立、版本化的 `DEPLOYMENT_POINT_ORDER`，建议使用按 `(row, column)` 的稳定行优先顺序：

```text
第1行：(1,1) (1,2) (1,3) (1,4) (1,5)
第2行：(2,1)       (2,3)       (2,5)
第3行：(3,1) (3,2)       (3,4) (3,5)
第4行：(4,1)       (4,3)       (4,5)
第5行：(5,1) (5,2) (5,3) (5,4) (5,5)
第6行：(6,1) (6,2) (6,3) (6,4) (6,5)
```

五个行营不进入 Pointer 输出空间。当前 `LAYOUT_ORDER` 是为旧“固定位置选棋子”接口设计的约束优先顺序，不能隐式充当新 Pointer 的位置编号。位置顺序版本必须随训练样本和检查点保存。

## 4. 部分布阵状态编码

### 4.1 状态定义

在第 $t$ 步部署当前棋子 $c_t$ 之前，模型输入为己方部分布阵：

$$
B_{t-1}=(b_{t-1,1},\ldots,b_{t-1,25}),
$$

其中每个位置的状态是 `EMPTY` 或已经部署的 12 类棋子之一。模型只能读取己方部分布阵、当前固定棋子、部署步编号和游戏模式，不能读取对手布局、后续随机结果或裁判隐藏信息。

固定序列和 $t$ 已经唯一决定剩余库存，因此不再单独输入 12 维剩余数量向量。

### 4.2 位置 token

对每个部署位置 $j$ 构造 256 维 token：

$$
x_{t,j}=E_{point}(j)+E_{occupant}(b_{t-1,j})
+E_{row}(j)+E_{point\_type}(j)+E_{mode}(m).
$$

- `E_point`：25 个稳定位置 ID；
- `E_occupant`：空位或已放置棋子身份；
- `E_row`：第 1～6 行；
- `E_point_type`：普通点或大本营等静态类型；
- `E_mode`：二人/四国模式。

将一个学习得到的 `[LAYOUT]` pooling token 与 25 个位置 token 一起送入 Pre-Norm Transformer：

$$
(h_{global},H_t)=\operatorname{LayoutEncoder}
([x_{layout},x_{t,1},\ldots,x_{t,25}]),
$$

$$
H_t=(h_{t,1},\ldots,h_{t,25}).
$$

位置编码器可以让 25 个点双向注意，因为 $B_{t-1}$ 只包含已经发生的部署，不含未来选择。这里不需要在 25 个空间点之间使用 causal mask；“自回归”来自每选一个位置就更新 $B_{t-1}\rightarrow B_t$，再计算下一步。

## 5. 当前棋子作为 Pointer Query

### 5.1 Query 构造

当前必须部署的棋子 $c_t$ 是 Query 的核心：

$$
e_t^{piece}=E_{piece}(c_t),
$$

$$
q_t=\operatorname{MLP}
([e_t^{piece};E_{step}(t);E_{mode}(m);h_{global}]).
$$

即使连续三步都是地雷，`E_step(t)`、已经更新的 $B_{t-1}$ 和 $h_{global}$ 也不同，因此三次位置分布不会被错误地当成同一个决定。

### 5.2 Pointer 打分

对 25 个当前位置分别计算缩放点积：

$$
s_{t,j}=\frac{(W_Qq_t)^T(W_Kh_{t,j})}{\sqrt{d_k}},
\qquad j=1,\ldots,25.
$$

因此：

```text
Q = 当前棋子 + 当前步骤 + 当前部分布阵的全局表示
K,V = 当前部分布阵的 25 个位置表示
输出 = 25 个位置 logits，而不是 12 类棋子 logits
```

主干仍采用 256 维、8 个注意力头、SwiGLU `d_ff=1024`、dropout 0。冷启动使用 8 层，主训练通过门槛后使用 16 层。网络名称统一为 `PieceConditionedLayoutPointerDecoder`。

## 6. Empty Mask 与 Piece Rule Mask

### 6.1 权威 Hard Mask

第 $t$ 步位置 $j$ 合法，当且仅当该位置尚未占用且满足当前棋子的静态规则：

$$
LegalMask_t(j)=EmptyMask_t(j)\land PieceRuleMask(c_t,j).
$$

具体规则为：

$$
PieceRuleMask(c_t,j)=
\begin{cases}
j\in HQ,&c_t=\text{军旗},\\
row(j)\in\{5,6\},&c_t=\text{地雷},\\
row(j)\ne1,&c_t=\text{炸弹},\\
\text{True},&c_t\text{ 是普通棋子}.
\end{cases}
$$

应用到 Pointer logits：

$$
\widetilde s_{t,j}=
\begin{cases}
s_{t,j},&LegalMask_t(j)=\text{True},\\
-\infty,&LegalMask_t(j)=\text{False}.
\end{cases}
$$

非法位置必须在 softmax **之前**设为 $-\infty$。不能先对 25 个位置归一化后再把非法概率清零，也不能采到非法位置后重试；这两种做法都会使记录的行为概率不正确。

### 6.2 规则不进入 loss

军旗、大本营、地雷后两行、炸弹非第一行、位置不可重复等都是生成空间约束，不是软惩罚：

- 不为非法部署设置负奖励；
- 不期待强化学习自己摸索规则；
- 不允许非法位置获得极小但非零的概率；
- 最终完整布局仍必须通过 `PlayerSetup` 再校验一次，形成双重防线。

## 7. 默认随机因子 0.7，并按最终概率部署

### 7.1 温度定义

本文把“随机因子”定义为 masked softmax 的温度，布局默认值固定为：

$$
T_{layout}=0.7.
$$

最终位置概率为：

$$
P_\phi(p_t=j\mid c_t,B_{t-1},m)
=\frac{\exp(\widetilde s_{t,j}/0.7)}
{\sum_{k\in\mathcal L_t}\exp(\widetilde s_{t,k}/0.7)}.
$$

`T=0.7` 相比 `T=1.0` 会让高分位置更集中，但只要合法集合中有多个有限 logit，分布仍然具有随机性。温度必须作为行为策略的一部分写入样本和检查点。

### 7.2 绝不使用 argmax

得到最终概率后必须执行：

$$
p_t\sim\operatorname{Categorical}
(P_\phi(\cdot\mid c_t,B_{t-1},m)).
$$

以下写法在训练、自对弈、联赛验证和正式部署中都禁止：

```python
position = probabilities.argmax()  # 禁止
```

正确语义是：高概率位置更常被选择，但低概率合法位置仍可能出现。不能在采样后又用“最高概率位置”覆盖采样结果。

### 7.3 随机种子与重复布局

同一模型、同一 seed 和同一运行环境得到相同布局是正确的确定性复现。生产 actor 必须让 RNG 状态持续前进，或按以下字段派生独立子流：

```text
experiment_seed, actor_id, episode_id, seat_id, layout_candidate_id
```

不能在每次布阵函数入口重新设置同一个固定 seed。随机采样并不保证任意两盘绝不重复；要求是概率分布不能退化为固定布局。强制去重会改变行为策略，如需采用必须精确建模条件采样概率，默认不采用。

## 8. 用 entropy 防止布阵坍缩

第 $t$ 步只在合法位置集合 $\mathcal L_t$ 上计算熵：

$$
H_t=-\sum_{j\in\mathcal L_t}P_t(j)\log P_t(j).
$$

布局总损失包含适度的负熵正则：

$$
L_L=L_{outcome}+\beta_LD_{KL}(P_\phi\|P_{ref})
-\alpha_L\frac1{25}\sum_{t=1}^{25}H_t.
$$

熵系数用于防止训练过早收敛到单一阵型，但不能大到压过终局胜负目标。更好的控制方式是监控合法支持集上的归一化熵并自适应调节 $\alpha_L$：

$$
H_{norm,t}=\frac{H_t}{\log|\mathcal L_t|},
\qquad |\mathcal L_t|>1.
$$

某一步只有一个合法空位时熵自然为 0，不应视为策略坍缩。默认随机机制是 `T=0.7 + Categorical sampling + entropy regularization`；不默认额外混入均匀噪声，只有消融确认需要时才增加概率下限。

## 9. 一盘游戏必须保存的布局轨迹

第 $t$ 步的训练动作是“为固定棋子 $c_t$ 选择位置 $p_t$”。每个座位必须保存：

| 字段 | 用途 |
|---|---|
| `piece_sequence_version` | 确认固定棋子顺序 |
| `point_order_version` | 解释 25 位 Pointer 下标 |
| `game_mode`、`seat_id` | 构造 mode token并归因终局奖励 |
| `piece_id[25]` | 审计实际使用的固定序列 |
| `position_index[25]` | learner 重放每步所选位置 |
| `legal_position_mask[25,25]` 或可重建的规则版本 | 保证新旧概率使用相同支持集 |
| `old_log_prob[25]` | 计算逐步重要性比率 |
| `entropy[25]` | 监控随机性和调节熵系数 |
| `temperature=0.7` | 精确重建行为分布 |
| `layout_model_version` | 检查样本陈旧度 |
| `rng_seed/state` | 审计和复现 |
| `terminal_reward` | 终局后写入的唯一环境回报 |

行为 log-prob 必须是对 Hard Mask 后、除以 `0.7` 并完成 softmax 的最终分布取值，不能记录原始 Pointer score，也不能只保存 25 步 log-prob 总和。

## 10. 游戏终局后反向传播布局参数

### 10.1 布局概率

完整布局概率为：

$$
P_\phi(p_{1:25}\mid C,m)=\prod_{t=1}^{25}
P_\phi(p_t\mid c_t,B_{t-1},m).
$$

游戏环境不可微。训练不穿过 `JunqiGame`，而是用终局结果形成布局 advantage，再乘以已采样位置的 log-prob。

### 10.2 逐 Pointer 步裁剪

生产训练从同一个空布局和对手上下文采样 $G_L$ 个候选，每个候选完成 $M$ 盘配对终局 rollout，得到组内标准化优势 $A^L_{bk}$。第 $t$ 个位置选择的比率为：

$$
\rho^L_{bkt}=\exp\left[
\log P_\phi(p_{bkt}\mid c_t,B_{bk,t-1},m)
-\log P_{old}(p_{bkt}\mid c_t,B_{bk,t-1},m)
\right].
$$

25 个 Pointer 决定共享同一个终局优势，但逐步裁剪：

$$
L_{L,GRPO}=-\frac{1}{BG_L\cdot25}
\sum_{b,k,t}\min\left[
\rho^L_{bkt}A^L_{bk},
\operatorname{clip}(\rho^L_{bkt},1-\epsilon_L,1+\epsilon_L)A^L_{bk}
\right].
$$

不能把 25 个比率相乘成一个序列比率。

### 10.3 严格更新时间

1. rollout 前冻结 `old_layout`；
2. `old_layout` 按固定棋子顺序和 `T=0.7` 采样位置，保存每一步最终 log-prob；
3. 用完整布局开始游戏，中间奖励始终为 0；
4. 游戏结束后，把该座位的 `+1/0/-1` 写回布局轨迹；
5. 收齐不可拆分的候选组后，用 `current_layout` 重算位置 log-prob、合法熵及对 `ref_layout` 的 KL；
6. 执行 `zero_grad()`、`loss.backward()`、梯度裁剪和布局优化器 `step()`；
7. 记录 loss、梯度范数、KL、熵、clip 比例及布局多样性。

终局之前禁止更新该轨迹。最小在线调试可以逐盘做 REINFORCE 更新；生产方案仍以完整候选组为原子边界，降低单盘胜负噪声。

## 11. 与走子模型共同训练

布局模型和走子模型使用两个优化器，不对离散环境做端到端求导。“共同训练”表示二者位于同一个自对弈闭环，使用相同规则和终局回报定义，并作为 `(layout, policy)` 成对检查点共同进化：

```text
固定棋子序列
    ↓
old_layout：每步 Pointer + Hard Mask + Softmax(T=0.7) + Categorical
    ↓ 完整布局及 25 步行为概率
JunqiGame ← old_policy 按信息状态概率采样走子
    ↓
终局 +1 / 0 / -1
    ├─ 策略根动作终局估值 → policy backward
    └─ 布局候选终局优势   → layout backward
```

主训练仍可采用“策略更新 8 个块、布局更新 1 个块”的交替节奏，避免两个模型同时快速漂移。8:1 是优化器频率，不表示布局模型脱离自对弈。控制所有座位的是同一个 Layout 实例；同步 rollout 期间该实例保持冻结，而不是为四个座位分别创建行为快照。

## 12. 布阵采样伪代码

```python
PIECE_SEQUENCE = (
    FLAG,
    MINE, MINE, MINE,
    BOMB, BOMB,
    COMMANDER, ARMY_COMMANDER,
    DIVISION_COMMANDER, DIVISION_COMMANDER,
    BRIGADE_COMMANDER, BRIGADE_COMMANDER,
    REGIMENT_COMMANDER, REGIMENT_COMMANDER,
    BATTALION_COMMANDER, BATTALION_COMMANDER,
    COMPANY_COMMANDER, COMPANY_COMMANDER, COMPANY_COMMANDER,
    PLATOON_COMMANDER, PLATOON_COMMANDER, PLATOON_COMMANDER,
    ENGINEER, ENGINEER, ENGINEER,
)

state = PiecePlacementBuilder(point_order=DEPLOYMENT_POINT_ORDER)
trace = []

for step, piece in enumerate(PIECE_SEQUENCE):
    partial_board = state.partial_board()
    point_states, global_state = layout_encoder(partial_board, mode)
    query = make_query(piece, step, mode, global_state)
    scores = pointer_scores(query, point_states)       # shape [25]

    legal_mask = state.legal_position_mask(piece)      # Empty AND PieceRule
    masked_scores = scores.masked_fill(~legal_mask, -inf)
    probabilities = softmax(masked_scores / 0.7)

    # 非常重要：按最终概率采样，不是 argmax。
    position = categorical_sample(probabilities, rng)
    trace.append((piece, position, legal_mask, log(probabilities[position])))
    state.place(piece, position)

setup = state.build_and_validate_with_player_setup()
return setup, trace
```

learner 在终局后使用 `piece_id[25] + position_index[25]` 重放部分布阵序列，重新建立 autograd 图；不需要让 actor 把计算图保留到整盘结束。

## 13. 验收标准

### 13.1 结构和规则

- 固定棋子序列长度为 25，计数与标准库存完全一致；
- 每一步网络输出恰好 25 个位置 score；
- 已占位置永远被 mask；
- 军旗仅有两个大本营可选，地雷仅第 5、6 行可选，炸弹不能选第 1 行；
- 一百万次采样非法布局率严格为 0，且每条轨迹都恰好填满 25 个不同位置；
- Pointer 结果能被 `PlayerSetup` 最终校验通过。

### 13.2 概率与随机性

- 默认布局温度严格为 `0.7`；
- 给定已知 logits 和 mask，概率与 `softmax(masked_logits / 0.7)` 数值一致；
- 采样调用使用 `Categorical`/multinomial，代码路径中不存在部署 `argmax`；
- 相同 seed 可逐步复现，相同持续 RNG 的多次采样能产生多个不同完整布局；
- 记录每步归一化熵、完整布局去重率、最常见布局频率、平均 Hamming 距离和两个大本营的军旗比例。

### 13.3 训练

- 游戏终局前布局参数不发生变化；
- 非零相对终局优势能产生有限、非零的布局梯度；
- actor 保存的 `old_log_prob` 能由 learner 在相同权重、状态、mask 和 `T=0.7` 下重放；
- 25 步共享终局优势但分别计算比率和裁剪；
- entropy 只在合法位置上计算，且不会压过终局 outcome loss；
- 布局与走子模型始终成对保存、验证和进入联赛。

## 14. 不可违反的设计约束

1. 固定的是棋子序列，网络输出的是位置；不能再实现成固定位置输出棋子类型。
2. 部署顺序必须从军旗、地雷、炸弹开始，再放普通棋子。
3. 数量由固定生成序列保证，不能放进 loss 让模型学习。
4. 合法性使用 `EmptyMask AND PieceRuleMask`，非法位置在 softmax 前设为 $-\infty$。
5. 默认随机因子/温度是 `0.7`。
6. 最终位置必须按 masked softmax 概率分类采样，绝对不能使用 `argmax`。
7. entropy 是适度的防坍缩正则，最终优化目标仍由真实终局胜负决定。
8. 布局训练样本必须走到终局，25 个位置选择共享该布局的终局优势。
