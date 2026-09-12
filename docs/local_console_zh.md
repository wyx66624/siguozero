# 本地训练监控与 CPU 对弈

服务地址为 <http://localhost:8765>，对弈室为 <http://localhost:8765/play>。
默认仅监听本机回环地址；Windows 通过 WSL 的 localhost 转发访问。关闭浏览器不会停止训练。

## 常驻服务与继续训练

监控已由 WSL 的 `siguozero-monitor.service` 托管，异常退出后等待 3 秒重启。
Windows 计划任务 `SiguoZero Monitor` 在当前用户登录时静默启动 WSL 和监控，
已取消任务的默认运行时限。关闭网页或终端后仍可访问监控。

首次安装或重新部署监控，在 WSL Ubuntu-24.04 中执行：

```bash
cd /mnt/e/agent/siguozero
sudo /root/anaconda3/envs/siguozero/bin/python tools/install_monitor_service.py
```

安装器先验证服务配置，再迁移该项目原有的监控进程；不会停止 GPU 训练。
重复安装相同配置时保留当前服务和 CPU 棋局；需要迁移时会检查是否还有未结束的棋局。
当前 WSL 已启用 systemd，安装不需要重启 WSL。

在 Windows PowerShell 注册登录启动任务：

```powershell
& E:\agent\siguozero\scripts\monitor_service_windows.ps1 -Register
```

计划任务执行隐藏的 PowerShell 入口，保持一个前台 WSL 调用，确保监控不依赖终端窗口。
这是因为 [WSL 官方文档](https://learn.microsoft.com/en-us/windows/wsl/systemd)
明确说明 systemd 服务本身不会维持 WSL 实例存活。
登录启动只启动监控；GPU 训练仍通过下面的明确命令启动或接续。

在 WSL Ubuntu-24.04 中，进入 `/mnt/e/agent/siguozero`：

```bash
# 启动/复用监控和对弈服务，不启动或恢复训练
/root/anaconda3/envs/siguozero/bin/python tools/local_console.py

# 明确启动本地四暗棋训练；已有同一任务时返回其 PID，不重复启动
/root/anaconda3/envs/siguozero/bin/python tools/local_console.py --start-training
```

训练配置为 `configs/local_4090_training.yaml`：沿用 96 局、8 个环境进程、
98304 环境步/轮、学习微批 32 的 4090 配置；总预算为 30 亿环境步。
与 `ppo_4090_throughput.yaml` 的区别仅为资源心跳每 15 秒一次，以及每 10 轮导出对弈快照。
这是本地四暗棋训练，输出到 `runs_local_4090/four_dark/with_dead_rules`。
旧双人训练目录保留原状。

启动器识别并复用已安装的 systemd 服务，检查 HTTP 端口与服务 PID 是否一致。
监控与 CPU 棋局属于同一个 systemd 控制组；GPU 训练在独立进程组中运行。
重启监控不会中断训练，但会结束当前 CPU 棋局。监控崩溃后，其 CPU 子进程也会回收。
若尚未安装服务，启动器仍支持原来的后台进程方式。

服务管理与日志（WSL 中）：

```bash
systemctl status siguozero-monitor.service --no-pager
journalctl -u siguozero-monitor.service -n 50 --no-pager
sudo systemctl restart siguozero-monitor.service
```

训练日志、PID 记录及 Windows 保活入口日志在 `output/local_console/`。
服务每次启动都会原子更新 `service.json`；`/api/health` 返回实际 PID 和管理服务名。
Windows 登录任务状态可用 `Get-ScheduledTask -TaskName 'SiguoZero Monitor'` 查询。
注册的登录入口已手动触发验证；为保持当前训练，没有通过重启电脑验证登录触发。
电脑关机、显式关闭 WSL 会停止当前训练；休眠期间训练暂停，服务不能跨关机保留进程内状态。

如需手动停止训练，先核对 `output/local_console/training-four-dark-local.json`
中的 PID 与命令，再向该 PID 发送 `SIGTERM`。训练器会在完整轮次边界退出。
服务本身不提供修改训练参数、暂停或删除检查点的 HTTP 接口。

## 监控口径

- 进程身份：同时核对 `/proc` 的命令、运行目录、启动时间与心跳 PID，排除 PID 被复用和旧日志误报。
- 存活、心跳过期、初始化、停止、预算完成分别显示。DDP 必须所有 rank 心跳正常才显示健康运行。
- PPO 用 `cumulative/environment_plies`；旧 GRPO 分支预算用 `cumulative/continuation_plies`。
- 完整周期吞吐用本次进程最近至多 7 条指标之间的累计步数差除以时间戳差。
  窗口内的保存、评测、布阵、同步、快照导出均进入分母；重启前的停机时段不混入。
- 轮内吞吐另列，不能代替完整周期吞吐。样本不足两条或进程停止时不展示当前 ETA。
- 20 天所需吞吐为“当前剩余步数 / 20 天”。这不是从首日开始固定的日历期限。
- 3B 步完整预算约需 1736 环境步/秒才能在 20 天内完成。
  之前性能测量外推约 27.3 天，实际工期还受长局、评测、保存与停机影响。
- 指标中的胜/和/负为训练自我对弈统计，不能据此宣称模型棋力提高。

## CPU 对弈与检查点选择

每个棋局启动一个独立 Python 进程，导入 torch 前清空 `CUDA_VISIBLE_DEVICES`，
同时清除分布式 rank 环境变量。模型始终以 `map_location="cpu"` 加载并运行于 CPU；
不加载优化器或 Critic 到设备，不建立 CUDA 上下文。
默认每局 2 个计算线程、最多 2 局，CPU 优先级降低；空闲 30 分钟后释放进程。
推理本身不使用显存，仍会使用本机 CPU 与内存。

界面显示三种权重来源：

1. **评测最优**：`model_selection/state.json` 中已完成评测的最优快照。
2. **最新训练快照**：本地训练开始时、之后每 10 轮原子导出的 Policy/Layout 权重。
3. **最近可恢复检查点**：训练器保存的 `checkpoints/latest.pt`。

未完成评测的初始基线、训练快照和性能实验模型均明确标注。
“最优”仅指本项目该模式、该实验的已完成评测；不代表外部最先进模型。
注册目录在 `configs/local_console.json` 中明确列出，不扫描并混用其他实验。

**旧双人主模型 update 212 是 v4，当前网络需要 v6**。
监控仍展示其历史进度，但对弈选择器禁用此不兼容项。
双明棋当前注册的是 v6 性能实验参考，不能视为经过棋力认证的冠军。
四暗棋新训练快照在初始化后即可选择；第 0 轮为未训练权重。

每局开局时复制参数到 CPU 模型，后续权重发布不改变本局对手。
页面刷新会恢复同一浏览器标签页中的棋局。刷新模型列表并重新开局可读取新版本。
支持座位选择、走法高亮、服务器合法性校验、回合版本校验、模型自动应手、
淘汰后继续观战、终局结算，以及仅含玩家可见信息的 JSON 棋谱导出。
棋盘和棋谱均通过规则引擎 `observe(human_seat)` 生成，暗棋隐藏身份不进入 API。

## 对弈快照与断点恢复不同

`runtime.inference_snapshot_every_updates` 默认 0，不改变其他训练配置。
本地配置设为 10；快照写到 `inference/policy_<update>_<generation>.pt`，
先完成文件，再原子发布 `inference/manifest.json`，保留最近三代。
导出复制 Policy/Layout 权重，不更改训练 RNG、参数或优化器。

沿用既有 **evaluation** 保存策略：每 5000 万环境步进行 100 局评测并保存训练状态。
**对弈快照不含优化器、Critic 和基础棋局池，不能作为断点恢复文件。**
首轮评测前若训练中断，该目录没有完整 `latest.pt` 时，启动器会拒绝从随机权重悄悄重启。
后续只能从真实存在的完整训练检查点恢复；不要将对弈快照重命名为 `latest.pt`。

## 验证

`tests/test_local_console.py` 覆盖进程存活与 PID 复用、完整周期计时、DDP 心跳缺失、
半行日志、格式兼容、检查点路径约束、三个模式的可见性、CPU 隔离、
合法/非法走法、旧回合请求拒绝、同局模型冻结、完整终局与座位旋转、同源访问和静态文件边界。
主仓库全量测试日志为 `output/local_console_full_tests.log`；服务专项测试日志为
`output/local_console_tests.log`。浏览器还验证了实际 v6 检查点开局和三席自动应手。

2026-09-11 的实际验证结果见
[运行证据](benchmarks/local_console_validation_20260911.json)：全量 271 项测试通过，
补充兼容性与进程退出检查后的服务专项 12 项通过。GPU 主训练在 update 10、
983040 环境步时，CPU 独立加载该轮自动快照并完成 12 步合法对弈；
加载与布阵 2.83 秒，三次应手分别为 0.047、0.132、0.135 秒，未初始化 CUDA。
同一时刻完整周期吞吐约 1294 环境步/秒，短窗口外推剩余 26.82 天；
后续长局和评测变化仍可能改变工期。第 10 轮快照导出耗时 0.644 秒。

常驻服务的新增验证见
[服务运行证据](benchmarks/monitor_service_validation_20260911.json)。
`tests/test_monitor_service.py` 检查安装归属、端口进程身份和重启后的 PID 发布；
本次 12 项监控/对弈测试和新增 4 项服务管理测试通过，日志为
`output/monitor_service_console_tests.log`、`output/monitor_service_unit_tests.log`。
实际强制终止监控后，服务用 3.29 秒恢复，测试 CPU 棋局子进程已回收。
GPU 训练 PID 1268 未改变，并从第 38 轮推进至第 39 轮。
`tools/verify_monitor_service.py --exercise-restart` 会在没有其他 CPU 棋局时创建测试棋局，
验证 CPU 推理和控制组隔离，再仅终止监控主进程，检查自动重启、CPU 子进程回收，
以及原 GPU 训练进程完成下一轮。该选项会造成几秒监控中断；不带选项则仅做只读检查。
