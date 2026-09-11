# 并行模型对弈与资源调优

训练中的“当前参数对最优参数”和独立历史模型评测共用并行调度器。默认每个 rank
推进 32 局，同一版本的 Policy 推理每批最多 32 条请求，使用 4 个 CPU 环境 worker。
候选和对手各加载一份 Policy + Layout，全部在该 rank 的设备上共享，不为每局复制模型。
二人、四暗、双明使用同一套资源配置，各自保留独立的规则、权重、结果目录和统计协议。

## 调度方式

- 每局保留独立棋盘、座位私有历史和采样随机数流；裁判把同一版本模型负责的观察
  合成推理批次。四国队友的私有观察仍然隔离，不合并为可互读的信息。
- 环境 worker 是同进程内的工作线程，处理各自对局的合法动作、行棋和历史推进，模型
  推理由调用线程批量执行。worker 不复制加速器模型。设置 1 个 worker 时直接在调用
  线程推进环境。纯 Python 环境计算仍受 GIL 限制，增加线程不等于 CPU 线性加速。
- 对局队列随结束随补入，不等待一批长局全部结束。多 rank 按完整两局换边对或四局
  轮转组分片，组内每局仍有固定种子，组完成后按原规则统计。
- 模型长历史前缀缓存按并发数预留：每个模型至少
  `max(temporal_cache_entries, parallel_games × 每模型控制座位数 × 3)` 个条目。
  默认二人每模型 96 个条目，四国每模型 192 个条目；`temporal_cache_entries: 8`
  是容量下限，不是所有并发局共用 8 个条目的上限。单局历史窗口滑动仍需重算必要的
  历史前缀，增加并发能把多个重算请求合批，但不能消除这部分计算。
- 棋盘编码缓存按活跃座位和历史窗口扩容，并按最近使用顺序淘汰旧记录，避免连续
  千局评测在缓存填满后一直保留最早几局。显存不足的重试会释放尚未交给缓存管理的
  中间 KV 页，避免多次重试累积占用。
- 加速器推理遇到显存不足时，调度器减半推理批量并重试；降到单条仍不足时明确失败，
  不把资源错误记为输棋。减小推理批量不减少比赛局数。

默认每轮自动选择仍为**所有 rank 合计 1,000 局**，不是每卡或每条并发队列 1,000 局。
得分、组级置信区间、完整棋谱及最优模型晋升规则不因并发而改变。独立随机数流使其他
对局的完成顺序不影响本局的随机数消耗；不同设备、算子或批量可能产生浮点误差，因此
不承诺跨硬件逐步棋谱完全一致。历史套件采用 `arena_version=3`，旧套件保留并另开输出目录。

## 训练入口

共享 YAML 的 `model_selection` 配置：

```yaml
model_selection:
  parallel_games: 32
  inference_batch_size: 32
  environment_workers: 4
  temporal_cache_entries: 8
```

三个训练模式都接受以下 CLI 参数：

```bash
python -m junqi.training.train_four_dark --model-scale main --device cuda \
  --arena-parallel-games 32 --arena-inference-batch 32 \
  --arena-environment-workers 4
```

`scripts/start_four_player_ppo.sh` 和 `scripts/train_npu_cluster.sh` 也读取 `EVAL_PARALLEL_GAMES`、
`EVAL_INFERENCE_BATCH_SIZE`、`EVAL_ENVIRONMENT_WORKERS`，默认分别为 `32 / 32 / 4`。
命令末尾显式传入的同名 CLI 参数优先。评测期间暂停本次训练的权重更新，完成后继续训练。
并发、批量和 worker 数可按当前资源调整，不改变训练进度或最优模型的保存位置。
`temporal_cache_entries` 配置下限仍属于已有训练选择记录的约定，续训时保持一致；
实际缓存容量会根据调整后的并发局数自动扩展。

## 独立历史评测

```bash
python -m junqi.training.evaluate_four_player --mode four_dark \
  --checkpoint-dir runs/four_dark/with_dead_rules/checkpoints \
  --baseline /path/to/older.pt --output-dir arena_parallel/four_dark \
  --device cuda --groups 250 --once \
  --parallel-games 32 --inference-batch-size 32 --environment-workers 4
```

二人入口为 `junqi.training.evaluate_two_player`，1,000 局使用 `--pairs 500`；
双明使用 `--mode double_open` 和对应双明 checkpoint。这里每个对手的局数为
`groups × 4` 或 `pairs × 2`，对手池有多个版本时需分别进行这些比赛。
历史套件固定参数和代码指纹；更换套件参数时应使用新输出目录。

NPU 专用脚本和 `EVAL_MODE=history` 的公共集群脚本读取：

```bash
EVAL_PARALLEL_GAMES=32 EVAL_INFERENCE_BATCH_SIZE=32 EVAL_ENVIRONMENT_WORKERS=4 \
EVAL_BASELINE=/shared/older.pt EVAL_OUTPUT_DIR=/shared/arena_parallel/four_dark \
NPROC_PER_NODE=2 bash scripts/evaluate_four_player_npu.sh \
  /shared/runs/four_dark/with_dead_rules/checkpoints/latest.pt --once
```

每卡一个 rank 时，并发局数和 worker 数都是**每卡**设置；总 CPU worker 数随 rank 数
增加。二人脚本为 `scripts/evaluate_two_player_npu.sh`，双明设置 `EVAL_GAME_MODE=double_open`。

## 调优与计时边界

默认 `32 / 32 / 4` 已在 RTX 4090 上进行容量验证：32 路满长 1001-token 合成历史，
另预留 3.5 GiB 模拟训练驻留内存，峰值约 16.8 GiB，未触发 OOM。它是合成容量探针，
不是实际训练进程保留全部运行状态时的显存上界；较小显存设备可先用 `16 / 16 / 4`。
实际运行仍需观察长局阶段的设备利用率、峰值显存、CPU 用量和实际局数/
秒。设备空闲且显存充足时可逐步增大并发和推理批量；显存紧张时先降低推理批量，再
降低并发。环境成为瓶颈时增加 worker，并确保 `rank 数 × worker 数` 适合可用 CPU
核心。小模型和短局的线程调度开销可能超过收益，CPU 调试可设为 `1 / 1 / 1`。

两种模式分别评测仍需各自完成比赛；可在不同空闲卡上同时启动，并分别指定设备与
输出目录。若共享同一张卡，两个进程的模型和历史缓存会同时占用显存，应分别下调
并发并重新计时，不能把单模式吞吐直接相加。

仅用 4 局短测外推无法反映大量长局、历史窗口滑动和模型加载的耗时。工程冒烟只能
验证调度与结果口径；正式耗时应使用相同模型、相同规则和种子，测量包含长局的实际
比赛并记录机器、并发配置、局数、总步数和耗时。完成这样的实测之前，不承诺固定
加速倍数或 1,000 局总时长。
