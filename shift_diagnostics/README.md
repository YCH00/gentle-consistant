# shift-diagnostics-v1 执行入口

当前交付的是阶段 A 的**资源发现工具和协议配置**。还没有完整的模型适配器、bank 采集器、固定 context 评估器或 B/C/D 执行程序，不能把审计成功当作实验验收通过。

完整规范见 `docs/SHIFT_ROBUSTNESS_EXPERIMENT_HANDOFF.md`；进度入口是仓库根目录 `STATUS.md`。

## 现在先做：服务器资源核验

1. 同步整个 `shift_diagnostics/`、根目录 `STATUS.md` 和交接文档。只同步这些新增材料，保留服务器实际训练源码；服务器和本地可能不同。
2. 激活成功运行主实验时的 Python 环境。审计使用 Python 3.7+ 标准库，不导入 torch/Gym/MuJoCo、不加载 pickle、不启动训练。
3. 在服务器项目目录执行：

```bash
bash shift_diagnostics/server_audit.bash
```

默认只尝试文档中的历史路径，并核验目录是否存在。实际路径不同时显式传入：

```bash
bash shift_diagnostics/server_audit.bash \
  /实际的/YinCH \
  /实际的/YinCH/GENTLE-consistant \
  /实际的/YinCH/shift-diagnostics-v1
```

其他模型在这个父目录之外时，直接使用 Python 入口追加多个 `--search-root`：

```bash
python shift_diagnostics/audit.py \
  --repo-root /实际项目目录 \
  --search-root /实际共享文件及各项目父目录 \
  --search-root /另一模型存储目录 \
  --hash-checkpoints \
  --out /新的诊断审计目录
```

这些 `/实际...` 是待替换的路径占位符。`--out` 必须不存在；重复核验生成新目录。默认搜索深度 9，可通过 `--max-depth` 增大。目录符号链接不自动跟随，外部链接的真实目标应显式加为 search root。排除旧测试产物、虚拟环境、缓存、凭据目录；候选模型只做文件哈希，不反序列化。其他 iteration 的显式文件名会被过滤；无 iteration 标记的 checkpoint 仅列为候选。

每次输出 `resource_audit.json`、`candidates.json`、`models.json`、`protocol.json`、`source_snapshots.json`、`STATUS.md`、实际命令 `commands.sh` 和 `audit_completion.json`。审计退出码 0 仅表示报告生成成功，`stage_A_complete` 仍为 false。

把这次审计目录中的报告同步回本地，尤其是前三个 JSON、源码快照与 STATUS。若服务器有本地缺失的 `evaluate_fixed_context*.py`、`shared_protocol/*.py` 或 baseline 适配源码，也同步其源码；不能仅凭候选列表推断实现可用。不要把模型大文件、许可证或私密凭据内容粘贴到聊天里。

## A 后半段：确认模型与共享契约，再做 smoke

- 56 个 evaluation 位置：7 方法 × 2 环境 × seeds 0–3 × iteration 499。GENTLE seed0/499 同时充当采集器；另外两个位置是各环境的 GENTLE seed0/200，共 58 个独立模型位置。
- 每个位置填实际路径、SHA256、训练配置、源码版本/工作区摘要、原生 context 字段、posterior、latent 维度、actor 接口、任务清单与 normalizer 哈希。分离 encoder/policy 要有同一 run/iteration 证据。
- `models.json` 当前是未确认清单，不自动按文件名选模型。相邻名为 GENTLE 的目录不能直接证明它是原版完整方法；同样不能将本仓库自动认作正式 Ours。
- 对共享任务清单、原始完整离线池及归一化验证格式和 provenance；任务参数按 manifest 恢复，不重新采样。比较训练使用的归一化和任务内容，而非只比较文件名。
- 根据确认后的源码逐方法实现适配器，严格加载权重，冻结参数与统计，保留原生 posterior。所有方法读取同一 raw smoke bank。每环境/方法做 1 task × 2 contexts × 1 rollout；物理环境 reset/step、有限奖励、动作范围、单次归一化、重复确定性表征、无参数变更均要有证据。
- 两个 MuJoCo 后端分别启动进程。`CUDA_VISIBLE_DEVICES=1` 时进程内部用 `cuda:0`。审计不能证明 MuJoCo 可运行。

只有上述验收通过才更新 A=COMPLETE 并进入 B。当前不提供尚未实现的 smoke 命令。

## 后续顺序与输出

| 阶段 | 工作 | 验收与产物 |
| --- | --- | --- |
| B | 唯一发布者采集并冻结共享 bank；所有完整方法读取同一 bank 做固定 context 评估 | 每环境 2,000 contexts，每 context 200 行；每方法/seed/环境 6,000 episodes；合计 336,000 episodes。输出 bank 哈希、逐 episode 回报、确定性 latent、seen/unseen 分表和行为变化图表 |
| C | 复用 B 的 policy499_sigma000 回报；独立采集 probe train/val context 与 query；冻结 encoder 训练诊断预测器 | 只用 seen 训练/验证；query 与 context 轨迹不交叉；相同 query 跨方法共享。H0 是无 z 的诊断参考，并非 RL 消融。输出逐任务 MSE/NMSE、训练日志与选择证据 |
| D | 复用 B latent，对固定 4 seen + 4 unseen 任务、2 条件、20 contexts 作图 | 每模型 320 点；按方法/seed 分别投影；输出 PDF/SVG/PNG、坐标、配置、Markdown/LaTeX 表格与解释边界 |

正式采集/评估命令将在 A 的实际接口确认后补齐，不能照抄不存在的旧入口。四种子统计使用 sample std（ddof=1）；先在同一 seed 内算配对差；不混合 seen/unseen 或不同环境 raw return。缺种子必须标记 incomplete。

## 每次运行和阶段结束都记录

在根 STATUS 和对应诊断输出目录 STATUS 中追加：时间、阶段、实际执行命令、退出码、输入/输出路径与哈希、验证结果、未完成项、下一步入口。实际命令存入 `commands.sh`，计划命令不能记为已执行。

```text
时间：
阶段 / 状态：IN_PROGRESS 或 COMPLETE（附验收证据）
实际命令 / 退出码：
输入版本与哈希：
输出目录：
验证结果：
未完成项：
下一步：
```

阶段 B/C/D 的输出组织遵循交接文档第 9 节。正式条件改变时使用新协议版本或新输出目录，并记录 protocol.deviations；恢复前核对协议、模型、数据和源码哈希。

本地检查审计工具：

```bash
python -m unittest discover -s shift_diagnostics -p test_audit.py -v
```

这些测试只验证审计工具，任何测试 fixture 都不能用于正式 smoke 或实验。
