# expand：512 维主干与集中式单值 Critic

本分支新增配置 `configs/expand_mappo_training.yaml`，用于四暗与双明的扩展模型。
本次只实现代码，未启动新训练、评测或测试用例。旧的 `local_4090_training.yaml` 仍用于原有训练。

## 模型结构

| 部分 | 原四国 main 配置 | expand 配置 |
| --- | --- | --- |
| action 单层投影 | 128 维 | 256 维 |
| state 整盘单层投影 | 128 维 | 256 维 |
| action/state concat | 256 维 | 512 维 |
| concat 后融合 | 直接进入时序主干 | 一层 SwiGLU，输入/输出均为 512 维 |
| Transformer 宽度 | 256 | 512 |
| 注意力头数 | 8 | 8 |
| 每个注意力头维度 | 32 | 64 |
| Actor 层数 | 32 | 16 |
| Critic 层数 | 32 | 16 |
| Critic 输出 | 胜负与和棋两个分量 | `Linear(512, 1)`，一个 V 值 |
| Transformer 内部 FFN 隐藏维度 | 1024 | 1024 |

“32 改为 64”按**每头维度**实现，实际头数保留 8。配置中的 `attention_heads` 表示头数。

融合层执行：

```text
action ── Linear → 256 ─┐
                       ├─ concat 512 → SwiGLU 512 → 加位置/局面元信息 → Transformer
state  ── Linear → 256 ─┘
```

SwiGLU 将输入分别投影成 512 维 gate 和 512 维 value，计算 `SiLU(gate) * value`，
再投影到 512 维。普通前向、数组批处理、增量 KV 推理和学习图共用这一融合规则。

Actor 与 Critic 使用相同的输入投影、融合模块、时序块结构，但参数、优化器和缓存独立。
Actor 与 Critic 均为 16 层。Critic 初始化时复制 Actor 的对应投影与全部 16 层；随后独立学习。
Actor 保留原有动作输出头。布阵网络与布阵奖励沿用现有设置。

## MAPPO 信息边界

沿用 PPO 的裁剪策略目标与 GAE，改为集中式价值学习：

- Actor：只读取 `GameHistory` 中该玩家可见的棋盘与历史；未知棋子仍是未知。
- Critic：读取独立 `CentralizedHistory` 中所有棋子的真实身份、真实盘面和公开规则状态。
- 环境进程分别返回私有观察和全知观察；两者使用独立数组和历史标识。
- Actor 的采样、似然计算、训练入口与学习图拒绝标记为全知的输入。
- 网页对弈和棋力评测只加载 Actor 与布阵网络，推理时不需要 Critic 或全知历史。

这遵循 MAPPO 的集中训练、分散执行思路；具体实现保留本项目的轮流行动、队伍胜负和历史对手机制。
方法参考：[The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)。
论文结果不等于本项目新架构的棋力或收敛速度验证。

## 单 V 的回报与 GAE

和棋对两队均为 `-0.15`，因此单个 V 不能在轮到对手时直接取负。
本实现对每个状态，用**同一个单输出网络**分别估计固定队伍 0 和固定队伍 1 的回报。
这是两次不同视角的估值，不是两个输出头。固定视角分别使用座位 0、1，视角座位被淘汰后仍然保留。

对每个队伍独立计算：

```text
delta[t] = reward[t] + gamma * (1 - terminal[t]) * V[t+1] - V[t]
A[t]     = delta[t]  + gamma * lambda * (1 - terminal[t]) * A[t+1]
```

随后为每个实际行动选择所属学习队伍的 advantage 和 V target：

- 军旗奖励保留：扛掉敌旗 `+0.2`，己方或队友旗被扛 `-0.2`，与终局奖励相加。
- 和棋惩罚对两队都为负；终局才切断 GAE，采样批次结束使用全知状态 bootstrap。
- 冻结历史模型的回合连接轨迹，但不产生它的 Actor 梯度。
- 现有重复送子惩罚仅记入实际行动的学习队伍，不转换成对方的正奖励。
- 布阵仍使用整局终局奖励。

每个真实 `game.step()` 只计一个环境步、保存一个训练样本。
两队的 Critic 前向估值不增加环境步数；现有 30 亿步预算口径不变。

## 配置与断点

新配置的关键字段：

```yaml
models:
  four_player:
    action_embedding_dim: 256
    board_embedding_dim: 256
    transition_token_dim: 512
    token_fusion_hidden_dim: 512
    temporal_layers: 16
    attention_heads: 8
    temporal_ffn_dim: 1024
ppo:
  algorithm: mappo
  critic_layers: 16
runtime:
  output_root: runs_expand_mappo
```

以后明确要开始训练时，可使用下面的命令；本次未执行：

```bash
python -m junqi.training.cli --mode four_dark \
  --config configs/expand_mappo_training.yaml --model-scale main --device cuda
```

默认新目录为 `runs_expand_mappo/four_dark/with_dead_rules`；双明使用自己的目录。
`--tiny` 仍缩小模型，仅供未来测试使用。二人模式保留 GRPO。

扩展模型写入 v10 断点，保存单 V 的 Critic、独立优化器、私有与全知历史以及 RNG。
恢复时检查算法、宽度、层数、注意力头数和融合配置。
旧 PPO 权重不能直接续训成新的 512 维 MAPPO；应建立新运行目录。
旧架构仍可读取支持的 v6–v9 权重，原配置继续写 v9。

新配置沿用现有采样规模及历史冠军评测日程。512 维网络的显存占用与吞吐没有实测，
原有 4090 性能数字不能直接用于本分支的工期预测。

## 验证范围

已进行静态 Python 语法、YAML 解析和差异检查，未执行项目代码、训练、评测或测试。
`tests/test_mappo.py` 已提供以下回归用例，留待运行验证：

- 配置维度、Actor/Critic 深度、独立参数和单个 V 输出；
- 隐藏敌子身份变化不影响 Actor 输入，全知输入拒绝进入 Actor；
- 工作者观察分离、滑动历史和全知历史序列化；
- 和棋、扛旗、冻结回合、淘汰跳过、终局重置和非终局 bootstrap；
- SwiGLU 普通、数组、打包与增量路径一致性及梯度；
- 环境步计数、训练池恢复、完整断点续训和 Actor-only 推理。
