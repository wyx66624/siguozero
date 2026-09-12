# SiguoZero 文档

本目录保存四国军棋与二人军棋的规则、棋盘编码、动作编码、可执行训练环境和自博弈方案。

**revision 22 当前四国编码**：棋盘和动作各 128 维，时序 token 256 维，main 32 层、FFN 1024。
缩小模型阶段的参数与历史并行情景见[128＋128 并行训练核算](compact128_parallel_training_eta_zh.md)。
实现及短轮诊断见[PPO 历史数组、批量学习和固定 KV 优化](ppo_pipeline_optimization_zh.md)：4090 的 2560 步诊断轮约 2.93 倍加速，包含数值、CUDA、分布式恢复验证。
上一阶段的 4 进程并行和约 70 天测量见[并行环境与学习端优化](ppo_parallel_optimization_zh.md)。
最新实现和工期见[20 天目标分项优化](ppo20_optimization_zh.md)：批量布阵、变长注意力、编译、融合优化器、批量 Critic 和分组在线流水。
48 局兼容配置实测外推 **33.5 天**，96 局大采样配置 **27.3 天**，均为连续训练时间，20 天目标尚未达到。
评测保存、停机和后期长局分布变化需另计。
此前串行环境版本的[48 局训练复测](four_dark_optimized_training_eta_zh.md)保留为对照，平均约 237 步/秒。
编码规则沿用[五维动作单层投影](action_linear_zh.md)与[整盘向量单层投影](whole_board_linear_zh.md)，二人宽度保持原值。
旧 512 维结构的时间估算保留在[revision 21 训练时间](action_linear_training_eta_zh.md)。
“为什么时间变长”、扩大 microbatch 的实测与人民币租卡预算见[动作与租卡费用核对](action_microbatch_rental_cost_zh.md)。
动作/棋盘完整码表见[输入编码](current_encoding_and_hardware_eta_zh.md)；该页参数和硬件时间表保留旧基线，最新情景以 revision 22 报告为准。

**四国当前训练入口**：[PPO、独立价值模型与 4090 显存实测](four_player_ppo_zh.md)。
revision 19 的[统一环境步定义](environment_step_budget_zh.md)：30 亿指训练采样与实际模拟分支的总交互预算。
预算口径见 [PPO 序列训练优化与 30 亿环境交互步预算](four_player_ppo_optimization_zh.md)；
修改前的[计算量与性能剖析](four_player_ppo_compute_audit_zh.md)说明棋盘编码和实现开销；
[30 亿总交互时间重估](four_player_ppo_eta_budget_zh.md)补测四暗、双明，并列出硬件条件情景和吞吐目标；
[显卡及多卡历史情景](four_player_ppo_hardware_eta_zh.md)未获目标硬件实测验证，已暂停用于排期；
[旧 PPO 计时](four_player_ppo_timing_zh.md)保留为未合并前缀的对照基线。
四国训练从 0 步建立基准、每 5000 万环境步进行 100 局选优，见[自动最优模型选择](best_model_selection_zh.md)，资源配置见
[并行模型对弈与资源调优](parallel_arena_zh.md)；固定旧对手比较见
[历史模型棋力评测](historical_arena_zh.md)。
自 revision 16 起四暗、双明使用 PPO；下列旧 GRPO 预算和架构 PDF 保留为历史
对照与二人模式说明，四国当前参数以这份 PPO 文档为准。

本地运行入口见 [训练监控与 CPU 对弈控制台](local_console_zh.md)：真实进程与心跳、
完整周期 ETA、固定版本人机对弈、训练快照发布和 WSL 启动方式。

建议按以下顺序阅读：

1. [四国军棋与二人军棋规则说明书](rules_zh.md)  
   定义棋子、棋盘、布阵、道路与铁路、行营、大本营、战斗、胜负及和棋规则。

2. [玩家主视角棋盘与路径编码规范](board_encoding_zh.md)  
   定义四国 129 点、二人 60 点的玩家相对编码，中央九宫旋转、底层路径和视角变换。

3. [四国军棋与二人军棋静态动作编码规范](action_encoding_zh.md)  
   定义 `(from_code,to_code)` 静态动作列表、固定不可能动作筛选、稳定动作编号和两阶段 Policy Head 映射。

4. [四国军棋与二人军棋自博弈游戏引擎](game_engine_zh.md)  
   说明布阵校验、行走、铁路寻路、战斗、轮转、观测、奖励以及训练环境接口。

5. [策略状态、棋盘快照与转移 Token 编码规范](state_token_encoding_zh.md)  
   定义玩家可见的全棋盘整数链、确定存活身份码与阵亡先验、四国 128＋128 / 二人 256＋256 的 concat token，以及固定初始 token 加最近 1,000 步的上下文。

6. [棋子条件自回归 Pointer 布阵模型与终局训练说明](layout_decoder_training_zh.md)  
   区分现有位置优先规则采样器与新版棋子优先 Pointer Decoder，说明固定棋子序列、25 点 Hard Mask、默认温度 0.7 的概率采样、终局反向传播及防坍缩要求。

7. [仅规则驱动的自对弈强化学习方案](reinforcement_learning_plan_zh.md)  
   定义布局 Decoder、棋盘编码器、512 维策略时序 Transformer、仅终局奖励的 Game-GRPO、联赛训练、超参数与验收门槛。经审核的启动值已落到 [configs/bootstrap.yaml](../configs/bootstrap.yaml)。

8. [训练计算量、显存、批大小与总样本预算](compute_budget_zh.md)  
   基于当前规则引擎随机对局实测，估算每锚点“旧策略采样 4 个根候选 × 每候选 2 条终局续局”的计算量、Actor/Learner 显存、Policy batch、训练总步数与硬件规模。

9. [三模式训练设置、资源需求与运行手册](training_resources_zh.md)  
   给出四暗、双明、二人三个独立训练/推理入口，K=4、M=2 参数、模型实际参数量、显存和耗时预算、WSL conda 环境、日志及自动断点恢复约定。

10. [死规则特征开关与消融训练](dead_rule_ablation_zh.md)  
    定义 `--dead-rules/--no-dead-rules` 的规则、输入和模型结构边界，以及两种变体的目录与检查点隔离。

11. [多 GPU 训练与历史编码加速设计](multi_gpu_and_performance_zh.md)  
    定义 torchrun/DDP 数据分片、每 rank 检查点、全局 batch 语义，以及已实现的增量 causal KV、持久 COW 历史、packed/ragged 输入、规则热路径和 RTX 4090 A/B 实测。

12. [模型架构与训练系统 PDF](../output/pdf/siguozero_model_architecture_zh.pdf)  
    用架构图汇总布局 Pointer、动作与棋盘 token、策略 Transformer、4×2 Game-GRPO、训练闭环、检查点及 CUDA 实测；其中架构图按默认带死规则版本绘制，关闭态差异见第 10 项。

## 文档关系

```text
规则说明书
    ↓ 决定棋盘拓扑与移动约束
棋盘与路径编码规范
    ↓ 提供点编码和底层线段
静态动作编码规范
    ↓ 提供训练与推理使用的动作编号
自博弈游戏引擎
    ↓ 生成动态合法动作掩码、状态转移与终局奖励
策略状态与转移 Token 编码规范
    ↓ 把玩家可见的动作—棋盘历史映射为模型输入
棋子条件 Pointer 布阵与终局训练说明
    ↓ 明确固定棋子顺序、位置采样、终局归因和布局参数更新
双模型自对弈强化学习方案
    ↓ 决定终局续局、锚点 batch 与资源需求
训练计算量与资源预算
    ↓ 可用同预算比较
带死规则 / 不带死规则结构消融
```

规则版本、棋盘编码版本、动作编码版本和 token 编码版本必须随训练样本及模型检查点一起保存，避免不同版本的点位、棋子码或动作编号混用。
