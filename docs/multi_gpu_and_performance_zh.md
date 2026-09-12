# 多 GPU/NPU 训练与历史编码加速设计

> revision 20 已改为[整盘向量单层投影](whole_board_linear_zh.md)。本文的逐点图编码器、旧参数量和已有性能数字属于修改前版本，不能作为新架构的计时或显存结论。

四国 revision 18 的 PPO 为 Policy/Critic 分别创建 DDP 包装和优化器；全局 batch
按真实动作数计，反传时顺序更新两个网络，采样时共享全座位的 Policy 和 Critic。
采样批与优化器小批分开：每 rank 每 512 个决策执行一次 optimizer step，
序列 microbatch=8 对完全相同的历史前缀共享计算并保留所有决策梯度。
当前 4090 容量及并行参数见[四国 PPO 优化报告](four_player_ppo_optimization_zh.md)。下文 `4×2`
分叉与旧 Actor 吞吐数字是 GRPO 口径。

## 1. 已实现的多加速器语义

训练采用 PyTorch `torchrun + DistributedDataParallel`（DDP），每张 CUDA GPU
或 Ascend NPU 启动一个进程。CUDA 使用 NCCL；910B 使用 HCCL，且支持 torchrun
多节点启动。它表示一套逻辑 Policy 的同步计算副本，并不表示多个玩家模型：

- 每个 rank 内仍只有一个 current Policy 和一个 current Layout；该 rank 上所有
  二人或四人座位都复用它们。
- 各 rank 独立推进一部分基础局和终局分支，绕开 Python GIL，并扩大环境并行度。
- Policy 梯度在 DDP rank 间同步；优化后所有 Policy 副本逐参数一致。
- Layout 终局样本汇总到 rank 0，rank 0 更新一次，再把 Layout 参数广播给所有 rank。
- KL reference 每个 rank 各有一份只读计算副本；它不是玩家模型。

`anchor_batch` 是全局 batch，必须能被 `WORLD_SIZE` 整除；
`policy_microbatch` 是每张卡的物理 microbatch。例如全局 `anchor_batch=128`、
`microbatch=24` 时：

| 卡数 | 每 rank 锚点 | 每 epoch、每 rank 的子批 |
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
  --anchor-batch 128 --microbatch 24 --actor-batch 192 \
  --rollout-anchor-wave 24 --temporal-cache-entries 576 \
  --target-continuation-plies 3000000000
```

四张 GPU 只需把 `--nproc-per-node` 改为 `4`。程序依据 `LOCAL_RANK` 绑定
`cuda:0...cuda:N-1`；NPU 则使用 `scripts/train_npu_cluster.sh` 绑定
`npu:0...npu:N-1`，完整单机/多机示例见根目录 README。不要为每个进程手工指定
不同 `--device`。

CPU 双进程冒烟测试：

```bash
torchrun --standalone --nproc-per-node=2 \
  -m junqi.training.train_two_player \
  --smoke-test --device cpu --anchor-batch 2 --base-game-pool 2 \
  --microbatch 1 --run-dir /tmp/siguozero-ddp-smoke --no-resume
```

多卡时不能安全地让单个 rank 在 CUDA/NPU OOM 后独自改变 microbatch，否则其他 rank
可能已经进入梯度 collective。实现选择 fail-fast：从最后一个完整原子检查点重启，
并显式调低 `--microbatch`。单卡仍保留自动减半重试。

## 3. 多 rank 检查点与日志

只有 rank 0 写 `latest.pt` 和归档，避免并发覆盖。保存前所有 rank 汇总各自的：

- 未结束基础局及各玩家历史窗口；
- 基础局 RNG、Python RNG、CPU RNG 和当前 CUDA/NPU RNG；
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

## 5. rev.15 已实现的等价加速

在不改变合法动作、采样语义、终局奖励或 learner 梯度定义的前提下，Policy/环境
增加以下复用：

1. **整状态去重**：同一个推理 batch 内完全相同的玩家历史只运行一次完整时序
   Transformer，结果按 inverse index 展开给各分支；
2. **棋盘 token 去重**：不同历史中的相同 `(mode, board_codes, casualty_bits)`
   只运行一次 BoardEncoder；
3. **冻结阶段跨步缓存**：一个 update 的 old-policy rollout 期间，把历史棋盘的
   256 维 global embedding 缓存在 GPU。后续步骤只重新编码新当前棋盘；进入
   learner/train 模式时自动清空，杜绝跨参数版本的陈旧 embedding。
4. **页式增量 causal KV**：冷启动前缀只完整前向一次；同一玩家下次行动时只为
   新增的 2/4 个 transition token 生成 K/V。32 层共用页表，完整页按引用计数共享，
   append 最多复制最后一个未满页。普通 learner 仍从原始 token 完整前向并建立
   autograd，不拿 rollout cache 反传。
5. **分支写时共享**：`PlayerHistory` 使用不可变持久链表。一个锚点的 8 个分支
   共享同一历史尾节点，追加动作仅创建一个新节点，不再复制最多 1,000 条记录。
   KV 状态也只在冻结 actor 中按前缀共享；所有座位始终调用同一 Policy 实例。
6. **packed/ragged 输入**：BoardEncoder 输入由 `[B,T,P]` 改为仅含有效历史 token
   的 `[N_valid,P]`；learner 按历史长度排序后再切 microbatch，降低时序 padding。
7. **规则热路径**：棋盘坐标变换、邻接边、道路/铁路邻居和路径类型改成静态查表；
   历史快照不再生成无用的合法动作 mask 和候选身份 mask。20 局、3,991 步的
   单核 profile 从 `5.70 s` 降到 `3.79 s`（约 `1.50x`）。
8. **单次 GPU→CPU 动作同步**：合法 source/destination mask 先批量建成 GPU
   tensor，source 与 destination 都在 GPU 采样，Python 裁判只接收最后一次
   `(source, destination)` 传输；续局不再回传未使用的 behavior log-prob。
9. **有界波调度**：4090 main 的实测甜点为每 24 个锚点（192 条终局分支）一波，
   actor batch `192`、LRU 前缀 `576`；波间释放页式 cache。峰值 CUDA 分配
   `16.66 GiB`，仍给 24GB 卡留出余量。

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
- `encoding/temporal_incremental_batches`
- `encoding/temporal_incremental_batch_mean`
- `encoding/temporal_incremental_batch_max`
- `encoding/temporal_attention_saved_fraction`
- `encoding/paged_kv_enabled`
- `encoding/paged_kv_pages`
- `encoding/paged_kv_peak_pages`
- `encoding/paged_kv_capacity_pages`
- `encoding/paged_kv_allocated_gib`
- `encoding/paged_kv_padding_fraction`
- `optimizer/temporal_padding_fraction`

双进程 CPU tiny 冒烟中，`board_encoder_saved_fraction` 为约 `54%`。这只证明
复用路径生效，不是 main 长局的性能承诺；正式收益必须用相同 checkpoint、种子、
局长分布和累计分叉步做 A/B profile。

## 6. 页式增量 KV 的实现边界

rev.15 已把一条增长历史的累计时序注意力从近似 `O(T³)` 降为 `O(T²)`。实现直接
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

- 物理存储已经是 16-token page 和引用计数 COW，不再逐 token `torch.cat` 整段
  历史；但每层 decode 仍先按页表 gather，再调用原生 SDPA。这还不是融合的
  FlashInfer/vLLM paged-attention kernel，因此仍有进一步消除 gather/kernel launch
  的空间。
- 绝大多数二人局在 1,000 token 前结束。达到滑窗边界后，固定初始 token 与滚动
  transition 的位置会在每次滑窗时变化；当前实现会在每次窗口滑动后保守地完整
  重建 cache，以保持与 learner 的绝对位置 embedding 语义一致。
- 增量 cache 只服务 frozen actor。Learner 必须从原始 token 重新前向，以保留
  完整 autograd，不能拿 detached rollout cache 反传。
- 随机页表 decode 与同一连续 K/V 的原生 SDPA 对照最大误差为 `0`；集成 tiny
  BF16 页式/旧连续增量路径 context 最大差约 `1.5e-3`，两者对完整 BF16 前向的
  舍入量级相同。合法 mask 与概率采样公式不变。

## 7. RTX 4090 实测（2026-09-03）

`tools/benchmark_rollout.py` 运行真实二人规则、K=4/M=2 分支和 main Policy。短局
固定相同种子做开关 A/B；自然终局探针还保留 current/reference Policy、
current/reference Layout 与两个 AdamW 状态：

| 场景 | 实现/批量 | continuation plies/s | 墙钟 | CUDA 峰值分配 |
|---|---:|---:|---:|---:|
| 8 锚点自然终局，旧连续 KV，batch 64/wave 8 | 434.29 | 51.41 s | 9.22 GiB |
| 同口径，仅去掉中途同步/无用 log-prob | 450.26 | 49.59 s | 约 9.2 GiB |
| 同口径，页式 KV，batch 64/wave 8 | 463.01 | 48.22 s | 10.78 GiB |
| 24 锚点自然终局，页式 KV，batch 192/wave 24 | **644.05** | 107.04 s | **16.73 GiB** |
| update 11：128 锚点、main 完整训练栈 | **651.97** | rollout 481.52 s | **16.66 GiB** |

update 11 从 update 10 的同一 checkpoint 独立恢复，产生 `1,024` 条 continuation、
`313,935` 步；完整 update 为 `488.86 s`，普通原子 latest 保存约 `6～8 s`。
旧配置 update 3～9 的加权 rollout 是 `481.37` 步/s、平均完整 update
`667.33 s`；因此新实现吞吐提高 `35.4%`，update 计算时间缩短 `26.7%`。
`microbatch=24` 与 actor batch `192` 均未回退，页式 arena 峰值使用
`5,876 / 14,400` pages。

以每 update 平均计入一次 latest 保存后的有效吞吐约 `633` 步/s，剩余
`2,996,804,192` 步约需 `54.8` 个连续运行日；按 90% 可用率是 `60.9` 日。
考虑后续分布变化和训练外评测，单张 4090 规划为 **61～70 天**，继续用至少
10 个新实现 update 的移动中位数修正。

2026-09-06 的长跑复核发现：旧实现按精确 prefix length 拆分页式 KV decode。
基础棋局池从 update 100 的 `6` 种历史长度扩散到 update 205 的 `52` 种后，
名义 actor batch `160` 实际退化成约 `8～10` row 的大量小调用。现在对不同长度
使用有效 token mask，并以完整 `1001`-token 历史窗合并 decode；每个独立 anchor
wave 还会归还已经无用的 SDPA 临时缓存。update 205 checkpoint 的完整训练栈、
全部 `64` 个池槽只读 A/B 达到 **500.38 continuation plies/s**，实际增量 batch
均值 `57.31`、最大值 `160`，无 batch 回退；峰值 reserved `19.81 GiB`，最后一个
wave 后为 `14.53 GiB`。这项数字是同断点 rollout 基准，仍需后续完整 update
持续确认。随后正式恢复的 update 206 完成 `196,778` 步，rollout 为
**509.90 continuation plies/s**，整轮从修复前 update 205 的 `1669.21 s`
降至 `413.86 s`；梯度、优化器和原子 checkpoint 均正常，并已继续 update 207。

## 8. 后续按收益排序的工程路线

1. **融合 paged-attention/CUDA Graph actor**：物理页式 COW 已完成；下一步消除
   page gather 和小 kernel 启动开销，只有实测快于当前原生 SDPA 才启用；
2. **原生批量规则环境**：把棋盘、子力、合法动作和战斗规则改为结构化数组，减少
   Python 对象 clone；优先 Rust/C++ 扩展或批量张量内核，而不是逐局 Python 循环；
3. **跨 update actor/learner 流水线**：当前只实现同一 rollout 内 CUDA 推理与
   CPU 环境 chunk 的安全重叠。真正跨 update 异步需要额外 behavior snapshot 或
   专用 actor GPU，并严格把 behavior version 限制为 1，单卡不能无代价重叠；
4. **profile 后再使用 compile/CUDA Graph**：动态 Python 对象和变长序列先稳定
   成静态张量接口，否则编译收益会被 graph break 抵消。

验收顺序应是：正确性与断点回放一致 → 每 rank 环境步吞吐 → GPU duty cycle →
端到端 `continuation_plies/s`。只看任务管理器瞬时 GPU 百分比不能判断训练速度。
