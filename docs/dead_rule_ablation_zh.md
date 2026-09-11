# 死规则特征开关与消融训练

四国 PPO 的独立 Critic 使用与 Policy 相同的 `dead_rules_enabled` 配置。
关闭时两个网络都删除阵亡特征输入与融合分支，检查点同时保存并校验该变体。

## 1. 一个开关控制整条数据链

训练入口提供互斥参数：

- `--dead-rules`：开启确定性死规则。
- `--no-dead-rules`：关闭确定性死规则。
- 两者都不写时读取 `configs/bootstrap.yaml` 中的 `runtime.dead_rules_enabled`；当前默认值为 `true`。

该开关不是“给相同模型喂全零先验”。它同时控制规则环境、历史记录和 Policy 架构：

| 环节 | 开启 | 关闭 |
|---|---|---|
| 确定性身份 | 持久记录由公开结果唯一推出的身份 | 不执行、不持久化 |
| 盘面整数码 | 可写入其他玩家的确定身份码 | 只保留信息模式本来公开的身份 |
| 阵亡先验 | 四暗 75 位、双明 50 位、二人 25 位，再散射到 75 位规范张量 | 历史记录为 `None`，不创建先验张量 |
| Policy 参数 | 有 `Linear(75, 256)` 和盘面融合层 | 两个模块均不存在 |
| 布局模型 | 不变 | 不变 |

关闭后仍保留军棋规则本身规定的公开信息：自己的棋子、双明中的友方棋子，以及司令阵亡后公开亮出的军旗。关闭的只是额外的确定性推导与阵亡库存特征，不会破坏裁判判定或合法动作生成。

## 2. 参数量差异

在 256 维正式架构中，开启态比关闭态多 `151,040` 个 Policy 参数；这些参数全部来自 75 维阵亡投影与盘面融合。Layout 参数量不受影响。

| 模型档位 | 开启态 Policy | 关闭态 Policy | Layout（两者相同） |
|---|---:|---:|---:|
| bootstrap | 26,610,696 | 26,459,656 | 8,887,296 |
| main | 144,157,704 | 144,006,664 | 17,292,288 |
| extended | 215,534,600 | 215,383,560 | 25,697,280 |

因此两种 Policy 的 `state_dict` 结构不同，不能互相载入。

## 3. 目录隔离

`--run-dir` 表示模式级基目录，训练器自动追加变体目录：

```text
runs/
└── two_player/
    ├── with_dead_rules/
    │   ├── checkpoints/
    │   ├── tensorboard/
    │   ├── metrics.jsonl
    │   └── train.log
    └── without_dead_rules/
        ├── checkpoints/
        ├── tensorboard/
        ├── metrics.jsonl
        └── train.log
```

四暗和双明采用同样结构。若传入的路径已经以正确变体名结尾，则不重复追加；若路径以相反变体名结尾，程序直接报错。

检查点格式记录 `dead_rules_enabled`，恢复训练和推理都校验该值。这样即使手工复制文件，也不能把开启态模型误续训到关闭态实验中。

## 4. 二人军棋启动示例

```bash
# 默认主实验：带死规则
python -m junqi.training.train_two_player \
  --config configs/bootstrap.yaml \
  --run-dir runs/two_player \
  --dead-rules

# 对照实验：不带死规则，也没有额外 25 位信息
python -m junqi.training.train_two_player \
  --config configs/bootstrap.yaml \
  --run-dir runs/two_player \
  --no-dead-rules
```

推理会自动从检查点读取变体；可额外写同名开关作为强校验：

```bash
python -m junqi.training.infer_two_player \
  --checkpoint runs/two_player/without_dead_rules/checkpoints/latest.pt \
  --no-dead-rules
```

建议正式比较时固定随机种子、模型档位、update 数、锚点 batch、`K=4, M=2` rollout 设置和评估对手池，仅改变本开关。
