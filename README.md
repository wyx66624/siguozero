# SiguoZero

面向四国军棋与二人军棋神经网络训练的规则、棋盘拓扑和玩家主视角编码基础库。

当前仓库先固定两类标准棋盘：

- 四国军棋：四个 30 点阵地加中央九宫，共 129 点。
- 二人军棋：两个 30 点阵地通过三条战线铁路连接，共 60 点。

完整文档索引见 [docs/README.md](docs/README.md)。其中规则口径与变体说明见
[docs/rules_zh.md](docs/rules_zh.md)，棋盘编码定义见
[docs/board_encoding_zh.md](docs/board_encoding_zh.md)，静态动作列表与 Policy Head
映射见 [docs/action_encoding_zh.md](docs/action_encoding_zh.md)，训练裁判接口见
[docs/game_engine_zh.md](docs/game_engine_zh.md)，策略输入的棋盘整数码和 512 维转移 token 见
[docs/state_token_encoding_zh.md](docs/state_token_encoding_zh.md)，仅依赖规则和终局结果的双模型自对弈方案见
[docs/reinforcement_learning_plan_zh.md](docs/reinforcement_learning_plan_zh.md)，棋子条件自回归 Pointer 布阵模型、默认温度 `0.7` 的概率采样与终局反向传播约定见
[docs/layout_decoder_training_zh.md](docs/layout_decoder_training_zh.md)，显存、batch、训练步数和总对局预算见
[docs/compute_budget_zh.md](docs/compute_budget_zh.md)，三种模式的完整训练设置、硬件需求和耗时估算见
[docs/training_resources_zh.md](docs/training_resources_zh.md)，经审核的冷启动
参数见 [configs/bootstrap.yaml](configs/bootstrap.yaml)。

## 快速使用

```python
from junqi.board import ArmPoint, FourPlayerBoard, FourPlayerSeat

board = FourPlayerBoard(viewer=FourPlayerSeat.SOUTH)

# 自己锋线左端：0
assert board.encode(ArmPoint(FourPlayerSeat.SOUTH, row=1, column=1)) == 0

# 对家同一局部位置：60
assert board.encode(ArmPoint(FourPlayerSeat.NORTH, row=1, column=1)) == 60

# 每条底层线段是 (起点编码, 终点编码)；无向版本总满足 start < end。
print(board.paths[:5])

# 图网络如需有向相邻边，可使用双向展开后的线段。
print(board.directed_paths[:5])

# 实际着法同样用有向端点对；长铁路着法可以跨越多个底层线段。
action = board.action(0, 127)

# 静态动作列表已删除包括工兵在内都永远不可能完成的点对。
assert len(board.actions) == 5624
action_id = board.action_index(*action)
assert board.decode_action(action_id) == action

# DeepNash 式两阶段动作头可使用压缩后的起点和条件目标槽。
space = board.action_space
origin_head_size = space.origin_count             # 121
destination_head_size = space.max_destination_count  # 76
destination_slot = space.destination_slot(*action)
```

创建一局可训练的四国军棋并执行动作：

```python
from junqi import JunqiGame

game = JunqiGame.new_four_player(seed=2026, max_plies=1000)

while not game.is_terminal:
    observation = game.observe()
    action_index = game.legal_action_indices()[0]  # 训练时替换为策略采样
    game.step(action_index)

print(game.result, game.rewards())
```

安装为可编辑包：

```powershell
python -m pip install -e .
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

## CUDA 自对弈训练与推理

每个训练模式只创建一套当前玩家网络：一个 `Policy` 实例和一个 `Layout`
实例。二人模式的两个座位、四国模式的四个座位都把各自旋转后的主视角状态送入
同一个实例，绝不会按座位启动 2 份或 4 份模型。布局一次批量采样所有座位，策略
状态则跨对局批处理。训练时另有一套只用于 KL 的 `reference` 模型对，但它不控制
任何玩家；同步 rollout 阶段直接冻结当前实例并保存旧 log-prob，不再深拷贝行为模型。

默认算法从冻结旧策略独立采样 `K=4` 个根路径，每个路径复制 `M=2` 次，后续每一步同样按旧策略概率分布采样直到终局。`--dead-rules` 开启持久确定性身份标注与阵亡先验；`--no-dead-rules` 同时关闭这些标注，并从 Policy 结构中彻底删除 75 维阵亡输入、投影层和融合层。默认值由 `runtime.dead_rules_enabled` 控制（当前为开启）。详细边界见 [docs/dead_rule_ablation_zh.md](docs/dead_rule_ablation_zh.md)。

三种模式和两种死规则变体都使用独立目录。即使传入同一个 `--run-dir` 基目录，程序也会自动追加 `with_dead_rules` 或 `without_dead_rules`；检查点固化该开关并拒绝交叉续训/推理。启动时默认从各自的 `latest.pt` 原子检查点恢复；`--no-resume` 遇到已有检查点会拒绝启动，确保不会误覆盖训练状态。

```bash
conda env create -f environment.yml
conda activate siguozero
pip install --no-deps --no-build-isolation -e .

python -m junqi.training.train_four_dark --run-dir runs/four_dark --dead-rules
python -m junqi.training.train_double_open --run-dir runs/double_open --dead-rules
python -m junqi.training.train_two_player --run-dir runs/two_player --dead-rules

# 无死规则消融；写入 runs/two_player/without_dead_rules
python -m junqi.training.train_two_player --run-dir runs/two_player --no-dead-rules

python -m junqi.training.infer_four_dark --checkpoint runs/four_dark/with_dead_rules/checkpoints/latest.pt
python -m junqi.training.infer_double_open --checkpoint runs/double_open/with_dead_rules/checkpoints/latest.pt
python -m junqi.training.infer_two_player --checkpoint runs/two_player/with_dead_rules/checkpoints/latest.pt
```

单卡安装验收可为任一训练入口添加 `--smoke-test --device cuda`；该开关会自动使用 tiny 测试模型。新运行会先保存 update 0，再进入 rollout；训练指标写入各变体目录内的追加式 `metrics.jsonl`、文本日志和可选 TensorBoard。独立资源心跳写入 `resource_metrics.jsonl` 并原子更新 `resource_latest.json`，即使一轮 rollout 很长也能看到当前阶段、PID、GPU 利用率/显存/温度/功耗、进程 CUDA 峰值和磁盘余量。可用 `--resource-monitor-seconds` 调整间隔，`--checkpoint-every`、`--archive-every` 和 `--keep-checkpoint-archives` 调整保存策略。检查点包含变体标记、两个模型、参考模型、两个优化器、训练计数、有效 microbatch、布局样本缓冲区、未结束基础局、玩家历史窗口和 CPU/CUDA/环境随机状态；只有开启死规则时才保存持久确定身份与确定阵亡库存。
