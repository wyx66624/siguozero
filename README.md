# SiguoZero

面向四国军棋与二人军棋神经网络训练的规则、棋盘拓扑和玩家主视角编码基础库。

四国模式（四暗、双明）当前采用 **PPO + 独立 Critic**，用 GAE 取代每步 8 条
蒙特卡洛终局续局。revision 20 已将棋盘编码改为**整盘向量经过一个线性层得到 256 维**，
移除逐点 Embedding 和棋盘 Transformer，详见[整盘编码与验证](docs/whole_board_linear_zh.md)。
revision 21 将动作改为**起点二维坐标、终点二维坐标、相对行动方 → Linear(5,256)**，
不再使用动作字段 Embedding 和战斗结果特征，详见[动作坐标编码历史](docs/action_linear_zh.md)。
revision 24 将和棋训练奖励改为 **-0.15**，并增加第六个动作输入：
**距 70 步无吃子和棋的剩余步数**。四国使用 `Linear(6,128)`，PPO 分别传播队伍胜负
与共同和棋惩罚，格式 6/7 断点可保留进度迁移到格式 8。详见[负奖励、步骤倒计时与迁移](docs/draw_penalty_countdown_zh.md)。
revision 25 增加**每人每局四次主动跳过**：动作 `0` / `(0,0)`，棋盘输入增加四个剩余次数，格式 9 从旧完整断点保留进度续训。详见[跳过动作与续训迁移](docs/pass_action_zh.md)。
参数和训练流程见[四国 PPO](docs/four_player_ppo_zh.md)。二人模式继续使用 Game-GRPO。
revision 22 将四暗、双明的棋盘和动作投影各缩为 **128 维**，拼接后 **256 维**，
main 保留 32 层，FFN 改为 **1024**；Policy 为 **36,324,226** 参数。
缩小模型阶段的参数、并行配置和历史硬件情景见[128＋128 并行训练核算](docs/compact128_parallel_training_eta_zh.md)。
PPO 已默认启用数组历史、批量学习和固定席位 KV/CUDA Graph；2560 步诊断轮实测约 **2.93 倍加速**，见[数据路径优化与回归验证](docs/ppo_pipeline_optimization_zh.md)。
四国 main 新增 **4 个常驻 CPU 环境进程、学习微批 32、直接因果 SDPA、关闭激活重算**，
此前完整训练和环境并行对照见[并行环境与学习端优化](docs/ppo_parallel_optimization_zh.md)。
最新一轮已加入**批量布阵、变长注意力、编译、融合 AdamW、延后批量 Critic 和分组在线流水**。
48 局兼容配置的 12 轮测量外推 **33.5 天**；另提供 96 局大采样配置，8 轮测量外推 **27.3 天**，
均为四暗 30 亿环境步连续训练，另计评测、保存、停机和后期长局变化。**20 天目标尚未达到**。
分项实现、数值边界、配置与完整计时见[20 天目标优化报告](docs/ppo20_optimization_zh.md)。
此前串行环境版本约 237 步/秒、147 天的测量保留在[上一轮训练时间](docs/four_dark_optimized_training_eta_zh.md)。
此前 512 维结构的计时与费用保留在[revision 21 时间核算](docs/action_linear_training_eta_zh.md)和[性能及费用核对](docs/action_microbatch_rental_cost_zh.md)。

当前仓库先固定两类标准棋盘：

- 四国军棋：四个 30 点阵地加中央九宫，共 129 点。
- 二人军棋：两个 30 点阵地通过三条战线铁路连接，共 60 点。

完整文档索引见 [docs/README.md](docs/README.md)。其中规则口径与变体说明见
[docs/rules_zh.md](docs/rules_zh.md)，棋盘编码定义见
[docs/board_encoding_zh.md](docs/board_encoding_zh.md)，静态动作列表与 Policy Head
映射见 [docs/action_encoding_zh.md](docs/action_encoding_zh.md)，训练裁判接口见
[docs/game_engine_zh.md](docs/game_engine_zh.md)，策略输入的棋盘整数码和按模式确定宽度的转移 token 见
[docs/state_token_encoding_zh.md](docs/state_token_encoding_zh.md)，仅依赖规则和终局结果的双模型自对弈方案见
[docs/reinforcement_learning_plan_zh.md](docs/reinforcement_learning_plan_zh.md)，棋子条件自回归 Pointer 布阵模型、默认温度 `0.7` 的概率采样与终局反向传播约定见
[docs/layout_decoder_training_zh.md](docs/layout_decoder_training_zh.md)，显存、batch、训练步数和总对局预算见
[docs/compute_budget_zh.md](docs/compute_budget_zh.md)，三种模式的完整训练设置、硬件需求和耗时估算见
[docs/training_resources_zh.md](docs/training_resources_zh.md)，经审核的冷启动
参数见 [configs/bootstrap.yaml](configs/bootstrap.yaml)。多 GPU/NPU 启动、全局 batch
语义和历史编码加速见 [docs/multi_gpu_and_performance_zh.md](docs/multi_gpu_and_performance_zh.md)。

## 快速使用

本地训练监控与 CPU 对弈控制台见 [使用说明](docs/local_console_zh.md)。
在 WSL 中运行 `/root/anaconda3/envs/siguozero/bin/python tools/local_console.py`，
打开 <http://localhost:8765>；加 `--start-training` 明确启动新的 4090 四暗棋训练。
对弈通过独立 CPU 进程加载冻结权重，监控进度和耗时来自实际训练日志与存活进程。

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
assert len(board.actions) == 5625
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

## 华为 Ascend 910B NPU 集群

`npu` 分支在保留 CUDA/CPU 路径的同时增加了 Ascend 910B 训练路径。每个
`torchrun` 进程绑定一张 NPU，模型梯度通过 HCCL 同步；未结束对局、随机状态和
布局样本等大型 Python 对象通过辅助 Gloo group 汇总到 rank 0 写检查点，避免调用
HCCL 不支持的 object gather。`anchor_batch` 始终是全局 batch，`microbatch` 是每张
NPU 的 learner batch。

### 固定环境与安装

该分支的 NPU 入口会在启动时校验下表，版本不匹配会直接退出，避免长任务在运行后
才暴露 ABI 或算子问题。

| 组件 | 支持版本 |
| --- | --- |
| Python | `3.12.x` |
| PyTorch | 固定 `2.7.1` |
| NumPy | `1.26.4`，且必须 `<2` |
| torch_npu | 默认 `2.7.1.post8`；安装包必须与 CANN 对应 |
| CANN | 默认 `9.1.0` |
| 训练精度 | 910B 默认 BF16；不可用时退到 FP16 + GradScaler |
| 分布式后端 | NPU tensor/梯度使用 HCCL，checkpoint object 使用 Gloo |

官方版本配套表给出的 PyTorch 2.7.1 对应关系如下。仓库默认采用当前支持
Python 3.12 二进制 wheel 的 `CANN 9.1.0 + torch-npu 2.7.1.post8` 组合。

| CANN | torch_npu 安装包版本 | 源码分支 |
| --- | --- | --- |
| 8.3.RC1 | `2.7.1`（PyPI 无 Python 3.12 wheel，需自行构建） | `v2.7.1-7.2.0` |
| 8.5.0 | `2.7.1.post2` | `v2.7.1-7.3.0` |
| 9.0.0 | `2.7.1.post4` | `v2.7.1-26.0.0` |
| 9.1.0 | `2.7.1.post8` | `v2.7.1-26.1.0` |

版本依据见 Ascend 官方维护的
[兼容性表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)和
[安装说明](https://github.com/Ascend/pytorch)。CANN、驱动和固件也必须按集群
型号配套，不能只替换 Python wheel。

兼容层只调用 PyTorch 2.7 已有接口：用 `torch.npu`/`torch.cuda` 完成设备、显存和
随机数管理，用 `torch.autocast` 与 `torch.amp.GradScaler` 完成混合精度；没有依赖
较新版本才提供的 `torch.accelerator` 门面。

```bash
conda env create -f environment-npu.yml
conda activate siguozero-npu
# CANN 9.1.0 默认路径；其他安装位置设置 CANN_ENV_FILE 给启动脚本
source /usr/local/Ascend/cann/set_env.sh

# x86_64 节点使用官方 CPU libtorch；aarch64 节点按 Ascend 文档安装 2.7.1
if [[ "$(uname -m)" == "x86_64" ]]; then
  python -m pip install 'torch==2.7.1+cpu' \
    --index-url https://download.pytorch.org/whl/cpu
else
  python -m pip install 'torch==2.7.1'
fi
python -m pip install 'torch-npu==2.7.1.post8'

python -m pip install --no-deps --no-build-isolation -e .
python -c "import sys, numpy, torch, torch_npu; print(sys.version); print(numpy.__version__); print(torch.__version__); print(torch_npu.__version__); print(torch.npu.device_count())"
```

若集群使用 CANN 8.5.0 或 9.0.0，把最后一条安装命令分别改为
`torch-npu==2.7.1.post2` 或 `torch-npu==2.7.1.post4`。CANN 8.3.RC1 的
`torch-npu==2.7.1` 未发布 Python 3.12 wheel；在本分支的固定 Python 版本约束下，
必须从 `v2.7.1-7.2.0` 分支自行构建 cp312 wheel。最终须同时满足 PyTorch 2.7.1、
Python 3.12、NumPy `<2` 和 CANN/torch_npu 配套关系。

### 训练模型

所有座位共享一套当前 `Policy + Layout`。四国 PPO 训练另有独立 Critic 和
Layout reference；二人 GRPO 训练另常驻冻结 Policy/Layout reference。
推理均只加载当前 Policy/Layout。

| 版本 | `MODE` | 训练入口 | 棋盘/信息模式 |
| --- | --- | --- | --- |
| 二人版 | `two_player` | `junqi.training.train_two_player` | 60 点、二人暗棋 |
| 四人版（四暗） | `four_dark` | `junqi.training.train_four_dark` | 129 点、四家身份均隐藏 |
| 四人版（双明） | `double_open` | `junqi.training.train_double_open` | 129 点、对家同盟信息公开 |

正式训练使用 `--model-scale main`：revision 21 带死规则的 Policy 为 `139,888,130` 参数，
Layout 为 `17,292,288` 参数，部署模型对合计 `157,180,418` 参数。四国 PPO 的
Critic 为 `139,526,657` 参数，三个可训练模型合计 `296,707,075` 参数。二人版与四人版
使用同一整盘线性编码和时序 Transformer 结构，但棋盘拓扑、合法动作空间和玩家主视角编码不同，因此
检查点不能跨模式加载。`bootstrap` 用于流水线验证，`extended` 用于后续放大实验。
新架构使用格式 6 检查点，旧图编码器或旧动作字段 Embedding 权重不兼容；切换架构时选择新的训练目录。

### 单机与多机训练脚本

单机 8 卡 910B（首次上机建议先把 `MICROBATCH` 保持为默认 `8`）：

```bash
NPROC_PER_NODE=8 RUN_DIR=/mnt/shared/siguozero-runs \
  bash scripts/train_npu_cluster.sh two_player

NPROC_PER_NODE=8 RUN_DIR=/mnt/shared/siguozero-runs \
  bash scripts/train_npu_cluster.sh four_dark

NPROC_PER_NODE=8 RUN_DIR=/mnt/shared/siguozero-runs \
  bash scripts/train_npu_cluster.sh double_open
```

两节点、每节点 8 卡时，两台机器使用相同的 `MASTER_ADDR`、`MASTER_PORT`、
`NNODES` 和共享存储 `RUN_DIR`，只改变 `NODE_RANK`：

```bash
# 节点 0
MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 NNODES=2 NODE_RANK=0 \
NPROC_PER_NODE=8 ANCHOR_BATCH=128 BASE_GAME_POOL=64 \
RUN_DIR=/mnt/shared/siguozero-runs \
  bash scripts/train_npu_cluster.sh two_player

# 节点 1
MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 NNODES=2 NODE_RANK=1 \
NPROC_PER_NODE=8 ANCHOR_BATCH=128 BASE_GAME_POOL=64 \
RUN_DIR=/mnt/shared/siguozero-runs \
  bash scripts/train_npu_cluster.sh two_player
```

`ANCHOR_BATCH` 必须能被 `NNODES * NPROC_PER_NODE` 整除；脚本默认每 rank 分配 8
个 anchor、4 个基础局。常用覆盖项还有 `MODEL_SCALE`、`MICROBATCH`、
`ACTOR_BATCH`、`ROLLOUT_ANCHOR_WAVE`、`TEMPORAL_CACHE_ENTRIES`、`UPDATES`、
`TARGET_CONTINUATION_PLIES` 和 `DEAD_RULES=0/1`。跨节点运行前应按集群网络设置
HCCL/Gloo 网卡并放通 rendezvous 与 HCCL 端口；不要让各节点使用彼此独立的同名
本地目录。

先做两卡最小闭环验收：

```bash
NPROC_PER_NODE=2 ANCHOR_BATCH=2 BASE_GAME_POOL=2 MICROBATCH=1 \
RUN_DIR=/mnt/shared/siguozero-smoke \
  bash scripts/train_npu_cluster.sh two_player \
  --smoke-test --updates 1 --max-game-plies 16 --checkpoint-every 1
```

容量和真实 rollout 探针也接受 NPU：

```bash
python tools/benchmark_cuda.py --device npu --mode two_player \
  --model-scale main --context-tokens 1001 --batch-size 8
python tools/benchmark_rollout.py --device npu --mode two_player \
  --model-scale main --anchors 8 --full-stack
```

### 评测脚本与指标

四暗、双明从初始模型作为基准开始，每累计 **5000 万次训练环境交互**选拔一次，
每次对阵此前最优模型 **100 局**（25 组四局轮转）；30 亿步共 60 次、6000 局。
二人保留进度 `30%、35%、…、100%`、每轮 1000 局的日程。
得分率超过 50% 时更新 `checkpoints/best.pt`。完整续训仍用 `latest.pt`；规则、配置与
棋力变化记录见 [训练中的自动最优模型选择](docs/best_model_selection_zh.md)。
四国完整续训断点每 **2500 万环境步**保存一次，初始化与正常退出也保存；评测仍每
5000 万步执行。本地对弈快照每 5 次更新生成。训练、评测和人机对局默认没有总步数上限，
连续 **70 步没有吃子**自动和棋。旧运行的完整迁移方式见[规则与断点迁移](docs/draw_rule_migration_zh.md)。

本地 4090 配置覆盖上述默认值：完整断点每 **1000 万步**保存，正常停止额外保存；
PPO 使用随训练进度收窄的优势自适应裁剪。监控区分已保存与未保存进度，详见
[动态裁剪与真实恢复进度](docs/adaptive_clipping_and_resume_zh.md)。

**验证新模型是否真的更强**：二人和四国使用独立的跨版本棋力评测入口、协议与成绩库：

| 模式 | Python 入口 | 每个对手的默认比赛 | 使用说明 |
| --- | --- | --- | --- |
| 二人 | `junqi.training.evaluate_two_player` | `--pairs 200`：200 对换边，400 局 | [二人评测](docs/two_player_arena_zh.md) |
| 四国（四暗 / 双明） | `junqi.training.evaluate_four_player --mode four_dark` 或 `--mode double_open` | `--groups 200`：200 组整队轮转，800 局 | [四国评测](docs/four_player_arena_zh.md) |

默认每增加 25 updates，候选模型对阵固定 baseline 和最近 2 个已评测模型；保存得分、
置信区间、和棋原因、公开棋谱与历史曲线数据。四国由新旧模型分别控制 0/2 与 1/3
整队，按队伍最终胜负计分；四暗、双明、二人的 checkpoint、对手池与输出目录不能混用。
支持 `--watch` 周期检查、`--once` 调度、断点续评和 CPU/CUDA/NPU 多 rank，
**不会启动或恢复训练**。原 `junqi.training.evaluate_history` 保留为二人兼容入口。
共用统计、安全和输出约定见 [历史模型评测说明](docs/historical_arena_zh.md)。以下原入口仍是同模型自对弈，不应混淆。

单 NPU 或集群评测会按全局 game index 分片，最终由 rank 0 汇总。它是同一检查点
控制所有座位的固定种子自对弈回归，适合检查训练退化、终局分布与吞吐；由于双方/各
方使用同一模型，`seat0/win_rate` 不能当作跨版本的绝对棋力分数。

```bash
# 单 NPU，输出汇总，并可选保存逐局记录
NPROC_PER_NODE=1 GAMES=100 EVAL_SUMMARY=eval/two_player.json \
EVAL_GAMES_JSONL=eval/two_player_games.jsonl \
  bash scripts/evaluate_npu_cluster.sh \
  runs_npu_910b/two_player/with_dead_rules/checkpoints/latest.pt

# 两节点 16 NPU；两端 NODE_RANK 写法与训练相同
MASTER_ADDR=10.0.0.10 MASTER_PORT=29501 NNODES=2 NODE_RANK=0 \
NPROC_PER_NODE=8 GAMES=1000 EVAL_SUMMARY=eval/four_dark.json \
  bash scripts/evaluate_npu_cluster.sh \
  runs_npu_910b/four_dark/with_dead_rules/checkpoints/latest.pt
```

主要指标及解释：

| 文件/指标 | 含义 |
| --- | --- |
| `metrics.jsonl` / `loss/policy_total`、`loss/layout_total` | Policy 与布阵模型总损失；关注趋势与 NaN/Inf，不直接比较两种模式的绝对值 |
| `policy/kl_reference` | 当前策略相对 reference 的精确 KL；配置目标 `0.015`，超过 `0.0225` 会提前停止本 epoch |
| `policy/approx_kl_old` | 四国 PPO 相对采样时旧策略的近似 KL，用于提前停止 |
| `loss/critic_total` / `critic/value_mse` | 四国 PPO 价值损失与价值预测误差 |
| `ppo/raw_nonzero_advantage_fraction` / `ppo/no_advantage_signal` | 四国 PPO 标准化前是否存在非零优势，避免把全零 loss 当成学会 |
| `critic/rollout_target_std` / `critic/rollout_explained_variance` | 采样时价值目标变化与 Critic 解释方差；目标方差近零时 EV 不输出，结合 `*_defined` 和 MSE 看 |
| `policy/clip_fraction` | PPO/GRPO ratio 被裁剪比例；达到 `0.30` 会提前停止本 epoch |
| `policy/entropy`、`layout/entropy` | 着法与布阵分布熵，用于发现过早塌缩 |
| `rollout/plies_per_second`、`rollout/continuations_per_second` | 全局 rollout 吞吐；应以目标 910B 集群实测建立基线 |
| `rollout/wins/draws/losses` | 当前 update 的终局续局结果分布 |
| `accelerator/*memory*_gib` | 每 rank 的 NPU/CUDA allocator 当前、保留和峰值显存 |
| 评测 `seat0/win_rate`、`draw_rate`、`loss_rate` | 0 号座位/队伍结果比例，三者之和为 1 |
| 评测 `plies/mean|min|max`、`terminal_reasons` | 平均/边界局长及终局原因分布 |
| 评测 `throughput/games_per_second`、`plies_per_second` | 整个评测集群按最慢 rank 墙钟计算的吞吐 |

训练会原子更新 `checkpoints/latest.pt`，并在 `metrics.jsonl`、`train.log`、可选
TensorBoard 和每 rank 的 `resource_metrics.rankNNN.jsonl` 中记录状态。恢复时可以
改变节点数/卡数，未结束基础局会重新分片；同一 launch 内所有 rank 必须使用一致的
模型、死规则开关和 batch 参数。

训练中判断是否学会，需结合学习信号与固定历史对手成绩。早期检查时机、轻量评测参数、
TensorBoard 查看方法及当前短测试的证据边界见[四国学习评估指南](docs/four_player_learning_evaluation_zh.md)。

## CUDA/NPU 自对弈训练与推理实现

每个训练模式只创建一套当前玩家网络：一个 `Policy` 实例和一个 `Layout`
实例。二人模式的两个座位、四国模式的四个座位都把各自旋转后的主视角状态送入
同一个实例，绝不会按座位启动 2 份或 4 份模型。布局一次批量采样所有座位，策略
状态则跨对局批处理。二人训练另有用于 KL 的 `reference` 模型对；四国 PPO
用独立 Critic 和 Layout reference。同步 rollout 阶段冻结当前实例并保存旧
log-prob，不再深拷贝行为模型。

四国 PPO 每次只执行一个真实动作，由 Critic 和 GAE 提供优势，再分别训练
Policy/Critic；使用完全相同前缀的序列训练、main microbatch=32，优化器小批仍为 512。
四国默认预算为 **30 亿训练环境交互**：每次执行游戏动作、推进一次状态计 1 步，
包含所有采样对局和实际执行的模拟分支；8 条分支各走 32 步计 256 步。
网络前后向和多轮 PPO 优化不增加计数。见[统一计数定义](docs/environment_step_budget_zh.md)。
当前 PPO 无额外蒙特卡洛分支，20 局并行、每轮采样 5120 步，对应预计 585,938 次更新；
并行配置与计数见[性能报告](docs/four_player_ppo_optimization_zh.md)，
两种四国模式的最新计时和单卡、多卡条件估算见[revision 21 时间重估](docs/action_linear_training_eta_zh.md)。
同一环境交互预算下蒙特卡洛与 PPO 的耗时比较见[算法成本口径](docs/four_player_mc_vs_ppo_zh.md)。
二人 GRPO 从旧策略采样 `K=4` 个
根动作，每个复制 `M=2` 次并运行到终局。`--dead-rules` 开启持久确定性身份标注
与阵亡先验；`--no-dead-rules` 同时关闭这些标注，并从 Policy/Critic 中删除
阵亡输入分支。详细边界见 [docs/dead_rule_ablation_zh.md](docs/dead_rule_ablation_zh.md)。

四国 RTX 4090 正式规格可使用 `bash scripts/start_four_player_ppo.sh four_dark`
或 `double_open`。旧 GRPO 权重迁移必须通过 `--init-from` 指定源检查点和新输出目录；
正常 PPO 续训会同时恢复 Critic 与优化器。见[迁移及运行说明](docs/four_player_ppo_zh.md#4-运行迁移和评测)。

rev.15 的 frozen actor 使用物理页式 causal KV、持久化 copy-on-write 历史、单次动作 GPU→CPU 同步和 packed 棋盘输入；4090 main 默认按 24 锚点/192 分支有界波运行。learner 仍对原始 token 完整前向并正常反传。可用 `--no-paged-kv` 或 `--no-incremental-inference` 做 A/B，也可用 `--temporal-cache-entries`、`--rollout-anchor-wave` 调整缓存。真实终局吞吐探针：

```bash
python tools/benchmark_rollout.py --mode two_player --model-scale main \
  --anchors 24 --max-game-plies 600 --actor-batch 192 \
  --anchor-wave 24 --temporal-cache-entries 576 --full-stack
```

三种模式和两种死规则变体都使用独立目录。即使传入同一个 `--run-dir` 基目录，程序也会自动追加 `with_dead_rules` 或 `without_dead_rules`；检查点固化该开关并拒绝交叉续训/推理。启动时默认从各自的 `latest.pt` 原子检查点恢复；`--no-resume` 遇到已有检查点会拒绝启动，确保不会误覆盖训练状态。

```bash
conda env create -f environment.yml
conda activate siguozero
pip install --no-deps --no-build-isolation -e .

python -m junqi.training.train_four_dark --run-dir runs/four_dark --dead-rules
python -m junqi.training.train_double_open --run-dir runs/double_open --dead-rules
python -m junqi.training.train_two_player --run-dir runs/two_player --dead-rules

# 两卡共同训练同一个二人模型；anchor-batch 是全局值，microbatch 是每卡值
torchrun --standalone --nproc-per-node=2 \
  -m junqi.training.train_two_player --run-dir runs/two_player \
  --dead-rules --anchor-batch 128 --microbatch 24

# 无死规则消融；写入 runs/two_player/without_dead_rules
python -m junqi.training.train_two_player --run-dir runs/two_player --no-dead-rules

python -m junqi.training.infer_four_dark --checkpoint runs/four_dark/with_dead_rules/checkpoints/latest.pt
python -m junqi.training.infer_double_open --checkpoint runs/double_open/with_dead_rules/checkpoints/latest.pt
python -m junqi.training.infer_two_player --checkpoint runs/two_player/with_dead_rules/checkpoints/latest.pt
```

单卡安装验收可为任一训练入口添加 `--smoke-test --device cuda`；双进程分布式验收可用 `torchrun --standalone --nproc-per-node=2 ... --smoke-test --device cpu --anchor-batch 2 --base-game-pool 2`。默认 `--checkpoint-policy periodic`，先保存 update 0，再按周期及正常退出保存；四国完整断点间隔为 2500 万环境步，二人默认每 5 次更新保存。训练指标写入各变体目录内的追加式 `metrics.jsonl`、文本日志和可选 TensorBoard。独立资源心跳写入 `resource_metrics.jsonl` 并原子更新 `resource_latest.json`，其他 rank 使用带 `.rankNNN` 的独立日志。可用 `--resource-monitor-seconds` 调整间隔；`--checkpoint-interval-environment-plies` 设置环境步保存间隔，显式 `--checkpoint-every` 改用 update 间隔。`--archive-every` 和 `--keep-checkpoint-archives` 控制周期归档。检查点包含模式/算法/变体标记、Policy/Layout（四国另含 Critic）、对应参考模型和优化器、训练计数、有效 microbatch、布局样本缓冲区、所有 rank 的未结束基础局、玩家历史窗口和 CPU/CUDA/环境随机状态；只有开启死规则时才保存持久确定身份与确定阵亡库存。
