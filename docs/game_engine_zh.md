# 四国军棋与二人军棋自博弈游戏引擎

版本：0.5  
规则版本：`rules_zh.md` 0.2  
棋盘编码版本：`board_encoding_zh.md` 0.2  
动作编码版本：`action_encoding_zh.md` 0.1

## 1. 目标与范围

本引擎提供一套无第三方运行时依赖、可复制、可复现的军棋裁判环境，用于：

- 四国军棋 2 对 2 自博弈；
- 二人布阵式暗棋自博弈；
- MCTS、策略价值网络和强化学习数据生成；
- 自定义合法布阵、随机合法布阵与中盘局面恢复；
- 四暗、双明、全明和二人暗棋/明棋观测。

实现文件：

- `src/junqi/pieces.py`：棋子、数量、军阶和布阵；
- `src/junqi/board.py`：棋盘拓扑、铁路方向和静态动作；
- `src/junqi/game.py`：合法行动、战斗、轮转、终局、观测与奖励。

## 2. 已实现规则

### 2.1 棋子与布阵

每名玩家严格使用 25 枚棋子：

| 棋子 | 数量 | 特性 |
|---|---:|---|
| 司令 | 1 | 最高军阶；阵亡后本方军旗公开 |
| 军长 | 1 | 普通军阶 8 |
| 师长 | 2 | 普通军阶 7 |
| 旅长 | 2 | 普通军阶 6 |
| 团长 | 2 | 普通军阶 5 |
| 营长 | 2 | 普通军阶 4 |
| 连长 | 3 | 普通军阶 3 |
| 排长 | 3 | 普通军阶 2 |
| 工兵 | 3 | 普通军阶 1；排雷并可在铁路转弯 |
| 炸弹 | 2 | 与敌子同归于尽 |
| 地雷 | 3 | 不能移动 |
| 军旗 | 1 | 不能移动；被夺取后玩家出局 |

`PlayerSetup` 对以下约束执行强校验：

1. 25 个非行营点必须全部且各放一子；
2. 5 个行营必须为空；
3. 棋子数量必须与标准库存完全相同；
4. 军旗必须位于两个大本营之一；
5. 地雷只能位于第 5、6 行；
6. 炸弹不能位于第 1 行。

随机布阵接受显式随机数生成器或种子，因此同一种子能够复现完全相同的初始裁判状态。

`LayoutBuilder` 可以按“固定位置、选择棋子”的旧方向逐点生成阵型。它采用“两个大本营、第一行、中间三行、第五行、第六行其余点”的固定 25 步位置顺序；每一步的 `legal_piece_types()` 不仅检查当前位置和剩余库存，还通过容量二分匹配确认余下棋子仍能完成合法布局。因此这个规则接口不会生成死前缀。训练包中的神经布局模型则采用“固定棋子、Pointer 选择位置”的相反方向，并在 `junqi.training.models` 内实现独立的棋子条件 25 位位置 Hard Mask，二者不要混用。

### 2.2 公路、行营与大本营

- 公路每手只能通过一条相邻印制线段；
- 行营可以进入和离开，但已有棋子的行营不能被攻击；
- 行营不是铁路点，从行营出发只能移动到相邻点；
- 任何棋子进入或开局位于大本营后都不能再移动；
- 大本营可以作为合法目标，但永远不能作为合法起点。

### 2.3 铁路

每条有向铁路边记录两个切线方向：离开起点的方向和到达终点的方向。

- 普通可移动棋子：相邻铁路边可以连续连接的条件，是上一段的到达方向等于下一段的离开方向。这样既允许直行，也允许沿棋盘印制弧线行进，但禁止在直角交叉点换线。
- 工兵：忽略上述方向连续条件，可以在铁路图中转弯。
- 所有棋子：不能越过任何占位点；有棋子的目标点可以成为路径终点，但不能成为中间点。

铁路搜索会把“到达点 + 到达方向”共同作为普通棋子的搜索状态，避免同一点从不同轨道方向抵达时被错误合并。

### 2.4 敌我限制

- 不能移动其他玩家的棋子；
- 不能进入或攻击己方棋子所在点；
- 四国军棋中不能进入或攻击对家盟友棋子所在点；
- 可以进入任何空点；
- 可以攻击不在行营中的敌方棋子。

### 2.5 战斗

| 情况 | 结果 |
|---|---|
| 高军阶攻击低军阶 | 进攻方留在目标点 |
| 低军阶攻击高军阶 | 进攻方移除，防守方保留 |
| 相同军阶相遇 | 双方移除 |
| 工兵攻击地雷 | 地雷移除，工兵留在目标点 |
| 普通棋子攻击地雷 | 进攻子移除，地雷保留 |
| 炸弹与任意敌子相遇 | 双方移除 |
| 可攻击棋子夺取军旗 | 旗方立即出局 |
| 炸弹攻击军旗 | 双方移除，旗方仍立即出局 |

司令以任何方式移除后，所属玩家的军旗在后续暗棋观测中公开。

### 2.6 回合、出局和终局

- 四国军棋按照 `SOUTH → EAST → NORTH → WEST` 的逆时针顺序轮转；
- 二人军棋双方交替行动；
- 已出局玩家自动跳过；
- 轮到一名玩家但其没有合法移动且跳过次数耗尽时，该玩家立即出局；
- 军旗被夺取时，该玩家出局，剩余棋子从棋盘移除；
- 四国一队的两名玩家均出局后，另一队获胜；
- 二人一方出局后，另一方获胜。

### 2.7 和棋

- 默认连续 70 手没有吃子时立即和棋；
- 战斗中任何棋子被移除会把计数器清零，包括进攻方被吃及同归于尽；
- 默认 `max_plies=None`，无总步数上限；诊断时可显式配置短局上限；
- 军旗导致的团队胜负优先于和棋计数。

## 3. 创建游戏

### 3.1 合法随机布阵

```python
from junqi import JunqiGame

four_game = JunqiGame.new_four_player(seed=2026)
two_game = JunqiGame.new_two_player(seed=2026)
```

同一个种子会产生相同布阵和相同初始 `state_key()`。

### 3.2 自定义布阵

```python
from junqi import GameConfig, JunqiGame, PlayerSetup

south = PlayerSetup.from_rows(south_rows)
north = PlayerSetup.from_rows(north_rows)

game = JunqiGame(
    GameConfig(variant="two_player", information_mode="dark"),
    setups=[south, north],
)
```

`south_rows` 和 `north_rows` 均为 6 行 × 5 列。五个行营必须填写 `None`，其他位置填写 `PieceType` 或其字符串值。任何非法布阵都会抛出 `SetupError`。

现有位置优先规则采样器的逐步接口：

```python
from junqi import LayoutBuilder

builder = LayoutBuilder()
while not builder.is_complete:
    coordinate = builder.next_coordinate
    legal_types = builder.legal_piece_types()
    selected_type = layout_policy.sample(coordinate, legal_types)
    builder.place(selected_type)

setup = builder.build()
```

这里的 `LayoutBuilder` 是无参数的旧方向规则约束状态机，不是神经网络；未显式传入 `setups` 时，基础规则环境仍可用随机采样器生成阵型。训练包已经实现的 `LayoutPolicyTransformer` 按照“军旗、地雷、炸弹、普通棋子”的固定棋子顺序，在当前部分布阵的 25 个位置上做 Hard Mask 和概率采样，并用 `PlayerSetup` 二次校验完整阵型；完整设计见[棋子条件自回归 Pointer 布阵模型与终局训练说明](layout_decoder_training_zh.md)。

### 3.3 恢复中盘裁判状态

```python
game = JunqiGame.from_position(
    config,
    pieces,
    current_player=0,
    active_players=(True, True),
    revealed_flags=(False, False),
    ply_count=120,
    no_interaction_plies=8,
    known_identities=per_viewer_live_exact_state,
    known_casualties=per_viewer_per_owner_dead_counts,
)
```

中盘恢复允许棋子数量少于完整库存，但会验证：

- 每种棋子不超过标准数量；
- 活跃玩家仍有且只有一面军旗；
- 军旗仍在所属玩家大本营；
- 地雷仍在所属玩家第 5、6 行；
- 若活跃玩家的司令已经不在棋盘上，其军旗必须标记为已公开；
- 已出局玩家不再保留棋子。
- 开启死规则时，每名观察者的确定阵亡计数不能超过该玩家真实已离场库存；亮旗方若已无司令，恢复时会自动固化“司令已阵亡”。关闭时恢复数据不得携带非空确定身份或阵亡表。

## 4. 自博弈接口

### 4.1 最小循环

```python
import random

from junqi import JunqiGame

rng = random.Random(7)
game = JunqiGame.new_four_player(seed=7, max_plies=1000)

trajectory = []
while not game.is_terminal:
    observation = game.observe()
    legal_indices = game.legal_action_indices()

    # 实际训练时由策略网络在 observation.legal_action_mask 上采样。
    action_index = rng.choice(legal_indices)
    trajectory.append((observation, action_index))
    transition = game.step(action_index)

final_rewards = game.rewards()
```

`step()` 同时接受：

- 静态动作列表下标；
- 当前玩家主视角的 `(from_code, to_code)`。

### 4.2 合法动作接口

```python
actions = game.legal_actions()          # 有向端点二元组
indices = game.legal_action_indices()   # 静态动作编号
mask = game.legal_action_mask()         # 与 board.actions 等长

# DeepNash 式两阶段动作头
origin_mask = game.legal_origin_mask()
start = actions[0][0]
destination_mask = game.legal_destination_mask(start)
```

联合动作掩码在四国固定长度为 `5625`，二人固定长度为 `1177`。两阶段起点掩码长度分别为 `121` 和 `56`，选定起点后的条件目标掩码固定长度分别为 `76` 和 `35`。掩码中为 `True` 的编号与对应的合法编号接口完全一致。

### 4.3 搜索树复制

```python
child = game.clone()
child.step(action_index)
```

`clone()` 复制所有可变裁判状态，但复用只读棋盘拓扑。对子节点的操作不会改变父节点。

### 4.4 状态键

```python
key = game.state_key()
```

状态键包含真实棋子身份，适合裁判状态缓存、调试和完全信息搜索。它不能直接作为暗棋玩家的网络输入，否则会泄漏隐藏信息。

## 5. 玩家观测

```python
observation = game.observe()        # 默认当前玩家
observation = game.observe(viewer)  # 指定玩家
```

`Observation` 包含：

- `points`：按该玩家主视角编码排列的棋子观测；
- `dead_rules_enabled`：本局是否启用确定性死规则特征；
- `casualty_players`：阵亡先验各行对应的相对玩家；四暗为 `(1,2,3)`、双明为 `(1,3)`、二人为 `(1,)`；
- `known_casualties`：上述玩家各自 25 位、与布局生成顺序一致的确定阵亡二值行；
- `current_player`：相对玩家编号；
- `active_players`：按自己、左敌、对家、右敌或自己、对手排列；
- `revealed_flags`：相同相对顺序下的军旗公开状态；
- `legal_action_mask`：只有当前行动玩家的观测具有非空合法掩码；
- `history`：重新编码到该玩家主视角的公开事件，默认保留最近 1,000 手；
- 当前手数、无交互计数和终局结果。

当 `GameConfig.dead_rules_enabled=false` 时，`casualty_players` 与 `known_casualties` 都是空元组，环境不持久化下述确定身份/阵亡推导。信息模式本来公开的身份和司令阵亡后的军旗公开仍正常生效，合法动作与裁判结算也完全不变。

未知棋子的 `ObservedPiece.kind` 为 `None`，但其占位、相对阵营、是否移动和 12 位 `candidate_mask` 仍然可见。候选身份只使用公开规则推断：移动后排除军旗和地雷；只有工兵才能完成的铁路转弯会把候选集合收缩为工兵；司令阵亡后军旗候选收缩为唯一军旗。候选集合始终包含真实身份，但不会把真实暗子身份用于额外筛选。

一旦身份能够被唯一确定，`ObservedPiece.kind` 和 `identity_visible` 会立即变成精确值，而不是要求策略模型永远从历史重新推理。环境还按观察者维护持久的 `known_identities` 与 `known_casualties[viewer][owner][piece_type]`。确定的存活身份随棋移动，阵亡后从存活表删除并写入对应死亡计数；这些状态都进入 `clone()`、`state_key()` 和训练中盘检查点，故截断 1,000 步公开历史或中断续训不会丢失结论。

战斗知识不依靠一组互相覆盖的手写 if，而是为每个观察者枚举所有仍合法的进攻/防守棋种对，并用公开战果、亮旗、夺旗、布阵位置、移动能力、既有确定身份及已耗尽库存筛选。只有所有剩余组合都同意某一方身份时才写入。由此统一覆盖：吃军长锁定司令、亮旗方吃师长锁定军长、工兵飞雷、司令撞雷、单方亮旗时司令与炸弹同归、双方亮旗时司令对碰、已知雷/旗与炸弹同归，以及未移动本阵第一排排除炸弹后的同级兑子。若结果仍有多个合法解释，环境保持未知。

军旗被扛或其他公开出局结算会移除该方全部棋子，因此所有观察者将该方规范 25 位阵亡行全部置 1。双明模式只输出两名敌方的 25 位行；自己与对家盟友的确定身份及阵亡知识同步共享。四暗不在盟友间广播私有结论。

这些结论严格按观察者可见信息计算。例如玩家 1 知道自己的军长被吃，可以认出进攻司令；在四暗模式中，不知道该守子原本是军长的玩家 2 不能获得相同结论。实现不得将裁判身份倒灌进 `known_identities`。

完整、不含隐藏身份的公开事件保存在 `game.public_history`。事件只记录行动者、物理起终点、移动或攻击、战斗结果、夺旗、公开军旗和出局玩家；`observe()` 会把这些字段转换为观察者的相对编码。传入 `history_limit=None` 可以取得完整历史，传入 `0` 可以省略历史。

| 模式 | 可见身份 |
|---|---|
| `four_dark` | 仅自己；已公开军旗除外 |
| `double_open` | 自己和对家盟友；已公开军旗除外 |
| `full_open` | 四家全部可见 |
| `dark` | 二人模式仅自己；已公开军旗除外 |
| `open` | 二人模式双方全部可见 |

表中的模式可见性是最低基线；开启死规则时，按上述规则已经确定的存活棋子也会对相应观察者显示精确身份。实际整数码、确定阵亡矩阵及关闭态契约见 `state_token_encoding_zh.md` 0.4。

## 6. 动作执行结果与奖励

`step()` 返回 `StepResult`，包括：

- 行动玩家与动作二元组；
- 进攻子和目标子；
- 移动、进攻方胜、防守方胜或同归于尽；
- 是否夺旗以及本手导致哪些玩家出局；
- 下一行动玩家、终局结果和当前奖励。

默认奖励是团队终局奖励：

```text
胜方每名玩家：+1
负方每名玩家：-1
和棋或未终局：0
```

训练端可以在不改变裁判逻辑的前提下，基于 `StepResult` 另行增加辅助奖励。

## 7. 异常边界

- 非法布阵：`SetupError`；
- 无效配置或恢复局面：`GameRuleError`；
- 当前局面中的非法动作：`IllegalActionError`；
- 终局后继续行动：`TerminalGameError`。

`is_legal_action()` 可用于无异常地预检一个动作。

## 8. 当前未包含的外围功能

核心裁判已经能够完整运行训练对局，但以下内容不属于本版本：

- 图形界面、网络联机和真人房间；
- 认输、求和协商、计时器与平台超时次数；
- 未在规则说明书中固定的重复局面规则；
- 翻棋、三人、六人或两人在四国完整棋盘上对角作战等变体；
- 神经网络、经验回放、MCTS 实现和训练调度器本身。

这些外围模块应调用本引擎，不应复制或改写裁判规则。

## 9. 验证范围

自动测试覆盖：

- 标准库存与全部布阵限制；
- 公路一步、行营、大本营和敌我占位；
- 普通棋子铁路直行、印制弧线及禁止直角转弯；
- 工兵铁路转弯与占位阻挡；
- 所有军阶、地雷、炸弹、军旗和司令公开规则；
- 四国逆时针轮转、盟友限制、玩家出局和团队胜负；
- 60 手无交互和棋与最大手数和棋；
- 暗棋、双明、全明观测和合法动作掩码；
- 隐藏身份交换不改变玩家观测、候选掩码和合法动作掩码；
- 公开事件的多视角旋转与工兵转弯公开推断；
- 候选组合唯一性推断、确定阵亡 25 位槽、工兵飞雷、司令/炸弹亮旗组合、首排同级兑子、吃军长/师长规则、双明共享、扛旗全阵亡，以及按观察者隔离和中盘恢复；
- 关闭死规则后不产生推断身份或阵亡行，同时仍保留公开亮旗、合法动作和裁判结算；
- 现有位置优先 `LayoutBuilder` 的逐点可完成性掩码；
- 克隆独立性、稳定状态键和随机自博弈压力运行。

### 主动跳过（2026-09-12）

`game.step(0)` 与 `game.step((0, 0))` 等价，均消耗当前玩家的一次跳过机会。其他同起终点仍非法，普通移动仍可使用棋盘点 0。`game.passes_remaining` 是绝对座位顺序；`Observation.passes_remaining` 按观察者相对座位排列。跳过事件为 `CombatOutcome.PASS`，公开物理起终点为 `None`，每位观察者读到的动作都是 `(0, 0)`。棋子、身份及阵亡知识不变。完整存盘、克隆与回放保留剩余次数。
