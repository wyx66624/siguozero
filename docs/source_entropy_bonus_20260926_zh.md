# 增加选子探索并从 update 5018 继续训练

2026-09-26，按用户指定公式，在原联合熵奖励外增加固定的 `0.005 × 选子熵`。

## 损失定义

当前策略为 `π(i,j|s) = p(i|s) q(j|s,i)`。使用自然对数，定义：

- `H_J = -Σ_ij π(i,j|s) ln π(i,j|s)`，所有合法完整动作的联合熵。
- `H_s = -Σ_i p(i|s) ln p(i|s)`，合法选子分布的熵。
- `H_d = Σ_i p(i|s) H(q(·|s,i))`，按当前选子概率加权的条件落点熵。

Actor 最小化：

```text
loss_actor = loss_PPO - mean(0.01 H_J + 0.005 H_s)
           = loss_PPO - mean(0.015 H_s + 0.01 H_d)
```

因此选子熵的有效系数从 `0.01` 增至 `0.015`；条件落点熵系数仍为 `0.01`。开局、后期使用同一固定值。损失使用原始熵，不使用前端显示的归一化熵。系数增加并不意味着选子概率、熵或守旗成功率会立即按比例增加。

## 实现与记录

`configs/expand_mappo_training.yaml` 的 `learner.entropy_coefficient.mode_overrides.four_dark` 新增 `source_coefficient: 0.005`。解析为 `TrainingSettings.source_entropy_coefficient`；未配置时默认为零。

普通 PPO 路径复用本次前向计算的选子概率，直接求完整选子熵；CUDA 图路径使用同样的掩码、FP32 归约和有效样本权重。新增计算不需要额外跑一次 Transformer。冻结对手不参与 Actor 熵梯度；只有一个合法选子类别时，该项为零；图中填充样本不计入损失。测试用的外部策略没有选子头时，从完整动作概率按起点求边缘分布。

新增指标：

| 指标 | 含义 |
| --- | --- |
| `policy/source_entropy_coefficient` | 新增系数，当前为 0.005 |
| `policy/source_entropy` | Actor 优化批次的原始选子熵 |
| `policy/conditional_destination_entropy` | `H_J - H_s` |
| `loss/policy_joint_entropy` | `-0.01 × H_J` |
| `loss/policy_source_entropy` | `-0.005 × H_s` |
| `loss/policy_entropy` | 上述两项之和 |

这些训练批次指标与前端已有的“更新后、战术分层抽样、近 20 轮选子归一化熵”使用不同样本和归一化口径，不能直接比较数值。

## 检查点与验证

原训练服务经 SIGTERM 完成 update 5018，保存 `reason=completed_or_stopped` 的 v17 检查点。累计环境步数为 `493,289,472`，保存了 128 个游戏槽、Actor/Critic/Layout、优化器和 Python/CPU/CUDA 随机数状态。已归档的 `update_000005018.pt` 保留修改前恢复点。

恢复沿用原服务及完整 `latest.pt`。序列化配置逐字段对比，仅新增选子熵系数及对应原始 YAML 字段；PPO、奖励、GAE、Critic、布局损失、学习率和历史对手设置未变。

验证通过：70 项 CPU 测试和 1 项实际 CUDA 图回放测试（动作头、熵及反向传播）。包括损失恒等式、源/目标独立头梯度、冻结样本、强制选子、填充行、普通/CUDA 图损失及梯度一致、捕获后的系数变更，以及恢复时模型/优化器/随机数/未完成棋局的完整保留。

复核材料保存在 `output/diagnostics/source_entropy_bonus_20260926_062731/`，包含修改前文件、仅本次改动的差异、检查点 SHA-256、检查点状态摘要和配置差异。

## 首轮实际运行结果

服务 PID `3837159` 从 update 5018 恢复，**update 5019 已完成**；监控 API 返回 `running`，继续采集下一轮。该轮包含 98,304 个 Actor 样本、3 个完整 epoch、288 次 Actor 优化器更新，累计环境步数 `493,387,776`。

| 实际指标 | 数值 |
| --- | ---: |
| `H_J` | 2.1505595073 |
| `H_s` | 0.8094686368 |
| `H_d` | 1.3410908704 |
| 联合熵损失贡献 | -0.0215055945 |
| 新增选子熵损失贡献 | -0.0040473431 |
| 熵损失合计 | -0.0255529377 |
| PPO surrogate 损失 | -0.0138466538 |
| Actor 总损失 | -0.0393995915 |

从运行日志独立重算 `-0.01 H_J - 0.005 H_s`，与记录误差约 `5.9e-10`；Actor 总损失误差约 `5.7e-10`。系数在开局、后期均符合配置，未发生 KL 提前停止。实际首轮 Actor CUDA 图占比为零，走普通打包训练路径；CUDA 图新增项的验证来自上面的单独捕获/回放测试。

完整数值见诊断目录中的 `first-completed-update.json`。本轮证据确认新目标已参与实际优化，不代表战术能力已经改善。
