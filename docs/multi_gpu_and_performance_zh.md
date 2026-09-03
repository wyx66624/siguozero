# 多 GPU 训练与历史编码加速设计

## 1. 已实现的多 GPU 语义

训练采用 PyTorch `torchrun + DistributedDataParallel`（DDP），单机每张 GPU
启动一个进程。它表示一套逻辑 Policy 的同步计算副本，并不表示多个玩家模型：

- 每个 rank 内仍只有一个 current Policy 和一个 current Layout；该 rank 上所有
  二人或四人座位都复用它们。
- 各 rank 独立推进一部分基础局和终局分支，绕开 Python GIL，并扩大环境并行度。
- Policy 梯度在 DDP rank 间同步；优化后所有 Policy 副本逐参数一致。
- Layout 终局样本汇总到 rank 0，rank 0 更新一次，再把 Layout 参数广播给所有 rank。
- KL reference 每个 rank 各有一份只读计算副本；它不是玩家模型。

`anchor_batch` 是全局 batch，必须能被 `WORLD_SIZE` 整除；
`policy_microbatch` 是每张 GPU 的物理 microbatch。例如全局 `anchor_batch=128`、
`microbatch=24` 时：

| GPU 数 | 每 rank 锚点 | 每 epoch、每 rank 的子批 |
|---:|---:|---:|
| 1 | 128 | `24+24+24+24+24+8` |
| 2 | 64 | `24+24+16` |
| 4 | 32 | `24+8` |

每个锚点仍产生 `K=4, M=2`，因此每次 update 的全局终局续局数始终是
`128 × 4 × 2 = 1,024`，不会随 GPU 数重复放大。

DDP 会在每张卡保留完整 current/reference 模型和优化器状态，所以它提高吞吐，
不会把多张卡显存自动合并成一个更大模型。如果单卡放不下模型本身，应另行引入
FSDP/ZeRO；如果只是 24GB 卡上的激活峰值过高，先降低每卡 `microbatch`。

## 2. 启动方式

两张 GPU：

```bash
torchrun --standalone --nproc-per-node=2 \
  -m junqi.training.train_two_player \
  --config configs/bootstrap.yaml \
  --model-scale main --dead-rules \
  --run-dir runs/two_player \
  --anchor-batch 128 --microbatch 24 --actor-batch 64 \
  --target-continuation-plies 3000000000
```

四张 GPU 只需把 `--nproc-per-node` 改为 `4`。程序依据 `LOCAL_RANK` 绑定
`cuda:0...cuda:N-1`；不要为每个进程手工指定不同 `--device`。

CPU 双进程冒烟测试：

```bash
torchrun --standalone --nproc-per-node=2 \
  -m junqi.training.train_two_player \
  --smoke-test --device cpu --anchor-batch 2 --base-game-pool 2 \
  --microbatch 1 --run-dir /tmp/siguozero-ddp-smoke --no-resume
```

多卡时不能安全地让单个 rank 在 CUDA OOM 后独自改变 microbatch，否则其他 rank
可能已经进入梯度 collective。实现选择 fail-fast：从最后一个完整原子检查点重启，
并显式调低 `--microbatch`。单卡仍保留自动减半重试。

## 3. 多 rank 检查点与日志

只有 rank 0 写 `latest.pt` 和归档，避免并发覆盖。保存前所有 rank 汇总各自的：

- 未结束基础局及各玩家历史窗口；
- 基础局 RNG、Python RNG、CPU RNG 和 CUDA RNG；
- 当前 rank 的环境分片。

共享模型、reference、优化器、累计全局步数和 Layout buffer 只保存一次。恢复时每个
rank 取回自己的环境和 RNG。旧单卡检查点第一次以多卡启动时，程序会把原基础局
无重复切成多个分片。相同 `WORLD_SIZE` 可精确恢复各 rank 随机流；如果卡数改变，
程序会合并再切分所有未完局，并按新 rank 确定性重置随机流，保留训练进度而不从头
开始，但不承诺与原卡数下逐动作完全相同。

rank 0 保持原日志名：`metrics.jsonl`、`train.log`、`resource_metrics.jsonl`。
其他 rank 使用 `train.rank001.log`、`resource_metrics.rank001.jsonl` 等文件；
TensorBoard 和聚合训练指标只由 rank 0 写。

## 4. 已定位的重复历史瓶颈

旧实现每次策略采样都把保留的完整历史重新送入模型。长度为 `T` 的一盘游戏中：

1. BoardEncoder 在第 `t` 步重复编码前 `t` 个棋盘，累计约为
   `1+2+...+T = O(T²)` 个棋盘编码；
2. 时序 Transformer 每步重新计算 `t × t` 因果注意力，累计复杂度接近
   `O(T³)`；
3. 同一锚点的八条终局分支共享绝大多数前缀，旧实现仍把这些前缀重复复制、
   collate 和编码；
4. Python 规则推进、合法动作生成、对象 clone 和逐步 GPU 调度使 GPU 呈脉冲式
   工作，显存未满也会出现低利用率。

这解释了为什么单纯换更快 GPU 或增大 learner microbatch 不能按峰值算力线性提速。

2026-09-03 对正在运行的旧单卡 main 进程采样：185 个 `terminal_rollouts`
资源心跳的 GPU 利用率平均约 `48.6%`，范围 `0%～89%`，进程 CPU 约 `113%`
（约一个繁忙核心），并且约 50 分钟后仍在 update 0 的终局采样。这与“Python
逐步规则调度 + 全历史重复计算导致 GPU 间歇等待”的判断一致。该进程启动后不会
热加载本次源码优化；需要在完整检查点边界重启，才会使用新缓存或 DDP。

## 5. rev.14 已实现的等价加速

在不改变合法动作、采样语义、终局奖励或 learner 梯度定义的前提下，Policy/环境
增加以下复用：

1. **整状态去重**：同一个推理 batch 内完全相同的玩家历史只运行一次完整时序
   Transformer，结果按 inverse index 展开给各分支；
2. **棋盘 token 去重**：不同历史中的相同 `(mode, board_codes, casualty_bits)`
   只运行一次 BoardEncoder；
3. **冻结阶段跨步缓存**：一个 update 的 old-policy rollout 期间，把历史棋盘的
   256 维 global embedding 缓存在 GPU。后续步骤只重新编码新当前棋盘；进入
   learner/train 模式时自动清空，杜绝跨参数版本的陈旧 embedding。
4. **增量 causal KV**：冷启动前缀只完整前向一次；同一玩家下次行动时只为新增的
   2/4 个 transition token 生成 K/V，新 query 读取旧前缀。普通 learner 仍从原始
   token 完整前向并建立 autograd，不拿 rollout cache 反传。
5. **分支写时共享**：`PlayerHistory` 使用不可变持久链表。一个锚点的 8 个分支
   共享同一历史尾节点，追加动作仅创建一个新节点，不再复制最多 1,000 条记录。
   KV 状态也只在冻结 actor 中按前缀共享；所有座位始终调用同一 Policy 实例。
6. **packed/ragged 输入**：BoardEncoder 输入由 `[B,T,P]` 改为仅含有效历史 token
   的 `[N_valid,P]`；learner 按历史长度排序后再切 microbatch，降低时序 padding。
7. **规则热路径**：棋盘坐标变换、邻接边、道路/铁路邻居和路径类型改成静态查表；
   历史快照不再生成无用的合法动作 mask 和候选身份 mask。20 局、3,991 步的
   单核 profile 从 `5.70 s` 降到 `3.79 s`（约 `1.50x`）。
8. **有界波调度**：每 8 个锚点（64 条终局分支）为一波，波间释放时序 cache；
   192 个 LRU 前缀是当前 4090 的实测甜点。128 个会抖动，384 个无额外命中却
   占用更多显存。

缓存上限由 `runtime.inference_board_cache_entries` 控制，默认每 rank `65,536`；
设为 `0` 可关闭。当前棋盘的 point embeddings 仍实时计算，只缓存历史所需的
global embedding，以免为每个棋盘常驻全部 60/129 个点向量而耗尽显存。

新增指标如下：

- `encoding/within_batch_history_saved_fraction`
- `encoding/within_batch_dedup_saved_fraction`
- `encoding/cross_step_cache_hit_fraction`
- `encoding/board_encoder_saved_fraction`
- `encoding/board_cache_entries`
- `encoding/temporal_cache_entries`
- `encoding/temporal_cache_hits`
- `encoding/temporal_cold_states`
- `encoding/temporal_incremental_tokens`
- `encoding/temporal_attention_saved_fraction`
- `optimizer/temporal_padding_fraction`

双进程 CPU tiny 冒烟中，`board_encoder_saved_fraction` 为约 `54%`。这只证明
复用路径生效，不是 main 长局的性能承诺；正式收益必须用相同 checkpoint、种子、
局长分布和累计分叉步做 A/B profile。

## 6. 增量 KV 的实现边界

rev.14 已把一条增长历史的累计时序注意力从近似 `O(T³)` 降为 `O(T²)`。实现直接
复用现有 `nn.MultiheadAttention` 的 Q/K/V 权重，增量 query 交给 PyTorch SDPA，
因此模型参数和旧 update-0 检查点兼容。结构如下：

```text
共享不可变前缀节点
        │
        ├── 分支 A：append 新 token → 每层只生成新 K/V
        ├── 分支 B：append 新 token → 每层只生成新 K/V
        └── 分支 C：复用相同 prefix page
```

当前约束：

- 每层新 token 的注意力是 `O(T)`；当前使用连续 K/V 张量，因为原生 SDPA 对连续
  张量明显快于在 Python 拼接许多小 page。波/LRU 提供有界内存和逻辑 COW，但这
  不是 vLLM 风格的自定义 paged-attention CUDA kernel。
- 绝大多数二人局在 1,000 token 前结束。达到滑窗边界后，固定初始 token 与滚动
  transition 的位置会在每次滑窗时变化；当前实现会在每次窗口滑动后保守地完整
  重建 cache，以保持与 learner 的绝对位置 embedding 语义一致。
- 增量 cache 只服务 frozen actor。Learner 必须从原始 token 重新前向，以保留
  完整 autograd，不能拿 detached rollout cache 反传。
- FP32 context 逐 token A/B 最大绝对误差约 `4.8e-7`；RTX 4090 BF16 四步累积
  最大绝对误差约 `0.003`，来自 fused attention 舍入次序。合法 mask 与概率采样
  公式不变。

## 7. RTX 4090 实测（2026-09-03）

`tools/benchmark_rollout.py` 运行真实二人规则、K=4/M=2 分支和 main Policy。短局
固定相同种子做开关 A/B；自然终局探针还保留 current/reference Policy、
current/reference Layout 与两个 AdamW 状态：

| 场景 | 增量 KV | continuation plies/s | 墙钟 | CUDA 峰值分配 |
|---|---:|---:|---:|---:|
| 4 锚点，最多 128 步 | 否 | 306.2 | 13.38 s | 0.70 GiB |
| 4 锚点，最多 128 步 | 是 | 475.0 | 8.62 s | 3.97 GiB |
| 4 锚点，最多 256 步 | 否 | 179.6 | 45.02 s | 0.80 GiB |
| 4 锚点，最多 256 步 | 是 | 435.4 | 18.57 s | 7.65 GiB |
| 8 锚点，自然终局/600 硬上限，完整训练栈 | 是 | **431.4** | 51.76 s | **9.22 GiB** |

自然终局样本共 `64` 条 continuation、`22,327` 步，平均 `348.9` 步；理论时序
attention pair 减少 `98.51%`。它仍是随机初始化模型的一次容量探针，不是长期
稳定吞吐保证。按该速度，30 亿 continuation plies 的纯 rollout 为 `80.5` 个连续
运行日；按 90% 可用率为 `89.4` 日。再为 learner、基础局、评测和检查点预留
`10%～25%`，单张 4090 当前应按约 **100～115 天（3.3～3.8 个月）**规划，首个
正式 update 后再用移动中位数修正。

## 8. 后续按收益排序的工程路线

1. **真正的 paged-attention/CUDA Graph actor**：消除当前 K/V `stack/cat` 和
   小 kernel 启动开销；只有实测快于连续 SDPA 才启用；
2. **原生批量规则环境**：把棋盘、子力、合法动作和战斗规则改为结构化数组，减少
   Python 对象 clone；优先 Rust/C++ 扩展或批量张量内核，而不是逐局 Python 循环；
3. **跨 update actor/learner 流水线**：当前只实现同一 rollout 内 CUDA 推理与
   CPU 环境 chunk 的安全重叠。真正跨 update 异步需要额外 behavior snapshot 或
   专用 actor GPU，并严格把 behavior version 限制为 1，单卡不能无代价重叠；
4. **profile 后再使用 compile/CUDA Graph**：动态 Python 对象和变长序列先稳定
   成静态张量接口，否则编译收益会被 graph break 抵消。

验收顺序应是：正确性与断点回放一致 → 每 rank 环境步吞吐 → GPU duty cycle →
端到端 `continuation_plies/s`。只看任务管理器瞬时 GPU 百分比不能判断训练速度。
