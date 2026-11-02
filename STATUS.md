# STATUS — shift-diagnostics-v1

更新：2026-11-02（Asia/Shanghai）。当前项目：`gentle-consistant`。

**当前：阶段 A 进行中。本地资源核验和服务器审计工具已完成；服务器资源绑定、模型适配和真实环境 smoke 未完成。B/C/D 尚未开始。** 用户负责将新增文件同步到服务器运行，再返回报告供下一步适配。

## 固定范围

- 仅比较 FOCAL、CORRO、CSRO、GENTLE、UNICORN-SS、ER-TRL、Ours 七种完整方法，不做组件消融。
- 环境：cheetah-vel、walker-rand-params；模型 seeds 0/1/2/3、iteration 499。
- 原版 GENTLE seed0 的 iteration 200/499 为唯一共享采集器；所有方法读取同一份冻结 raw context bank。
- H=200、K=20、L=3；每 rollout 最多 200 步，固定 context，不追加反馈，不沿用累计 600 步的主实验评估。
- 交接文档：`docs/SHIFT_ROBUSTNESS_EXPERIMENT_HANDOFF.md`，与用户提供文件 SHA256 相同：`4b16fa5dc8ffdd1daa7b0a8feb56e38040f22c40039f808bfef09766ebf8a850`。

## 阶段状态与下一步

| 阶段 | 当前状态 | 进入下一阶段的条件 |
| --- | --- | --- |
| A：核验与适配 | IN_PROGRESS：本地核验完成；服务器审计待执行 | 完成资源绑定与原生适配；每方法/环境使用同一 raw context 的真实 smoke 通过 |
| B：bank 与实验 1 | NOT_STARTED | 每环境 2,000 contexts；每方法/seed/环境 6,000 episodes，四种子完整；回报/latent/哈希可核验 |
| C：实验 2 | NOT_STARTED | 独立 probe 数据无轨迹交叉；seen-only 训练/验证；同一 query 跨方法共享 |
| D：实验 3 与论文材料 | NOT_STARTED | 保存 latent/投影配置可重画；逐样本结果可重算表格 |

## 已核验资源与限制

- 本地 Git HEAD：`6e44eecb5427e57a8ac3453858d53f3f418f6908`。任务开始前 `.gitignore` 已修改，本轮没有改动该文件或训练代码。
- 当前目录及已检查父目录未发现适用的 AGENTS.md。
- 有 `train_gentle.py`、Agent/actor/环境实现和两个环境配置；静态接口证据在 `shift_diagnostics/ADAPTER_NOTES.md`。
- 缺少 `evaluate_fixed_context.py`、`evaluate_fixed_context_suite.py`；`shared_protocol` 当前仅缓存、无 Python 源码。
- 排除旧测试产物后，本次当前仓库扫描未发现 checkpoint 候选或真实共享 task_manifest/normalizer。扫描范围、深度和排除项保存在 resource_audit；这不等于证明服务器缺失。
- 本地相邻目录存在其他 baseline 源码入口和多个 GENTLE 版本，但尚未验证对应正式 run；不能将目录名视为模型身份。
- 当前本地 Python 为 3.12.3；包元数据未找到 numpy/torch/gym/mujoco-py/sklearn。本轮没有运行真实环境或加载模型。
- 本地环境配置中的历史 data_dir 不一致，未擅自修改；应以服务器成功主运行的共享文件和配置为准。

## 本轮新增文件

- `shift_diagnostics/audit.py`：只读资源发现、候选文件哈希、源码摘要、58 个未确认模型位置、运行记录和审计完成标记。不会自动把候选提升为正式模型。
- `shift_diagnostics/protocol.json`：固定协议；共享资源绑定待核验，deviations 为空。
- `shift_diagnostics/server_audit.bash`：服务器一条命令生成独立审计目录。
- `shift_diagnostics/README.md`：A→B→C→D 顺序、当前可执行命令、每阶段记录模板。
- `shift_diagnostics/ADAPTER_NOTES.md`：已有源码接口和适配注意点。
- `shift_diagnostics/test_audit.py`：审计护栏检查，不是实验 smoke。

## 实际运行记录

工作目录均为 `D:/Desktop/offline meta RL/gentle-consistant`。

1. 本地源码与资源读取：读取交接文档、训练配置、Agent、actor、normalizer、环境包装和注册；`git -c safe.directory='D:/Desktop/offline meta RL/gentle-consistant' rev-parse HEAD` 核验版本。Git 信任只在单条命令生效，未修改全局配置。
2. `python -m unittest discover -s shift_diagnostics -p test_audit.py -v`：第一次退出码 0，3 项通过（完整模型矩阵、协议与交接文档哈希、候选/排除/不反序列化/不覆盖/报告哈希）。后续最终复验结果追加于下方。
3. `python shift_diagnostics/audit.py --repo-root . --hash-checkpoints --out output/shift-diagnostics-v1/audits/local-A-20261102-01`：退出码 1；本地旧 output 目录 WinError 5 写权限拒绝，没有完成报告。未更改 ACL，改用项目中新目录。
4. `python shift_diagnostics/audit.py --repo-root . --hash-checkpoints --out diagnostics_audits/local-A-20261102-01`：退出码 0，资源发现完成；checkpoint 候选 0，shared Python 源码 0，scan_errors 为空。报告中的 A_complete=false。此报告保留其生成时的源码哈希；最终工具修订后使用新审计目录复核。
5. 最终复验 `python -m unittest discover -s shift_diagnostics -p test_audit.py -v`：退出码 0，3/3 通过。所有新增 Python 文件通过 `ast.parse(..., feature_version=(3,7))` 语法检查；这不是 Python 3.7 运行时实测。`C:/Program Files/Git/bin/bash.exe -n shift_diagnostics/server_audit.bash` 退出码 0。
6. `python shift_diagnostics/audit.py --repo-root . --hash-checkpoints --out diagnostics_audits/local-A-20261102-02`：退出码 0。**当前有效本地审计目录为此路径**；checkpoint 候选 0、shared Python 源码 0、scan_errors 为空；两个候选仅为 train_gentle.py 和本轮拟定 protocol.json，不是正式共享数据。
7. 使用 Python hashlib 对 `local-A-20261102-02/audit_completion.json` 中 7 个输出文件逐个复核 SHA256，全部通过；交接文档 SHA256 与协议匹配，audit.py 当前源码与报告一致。A 的真实模型/环境 smoke 仍未执行。

## 服务器下一条命令（计划，尚未执行）

同步 `shift_diagnostics/`、本 STATUS 和交接文档；不要用整个本地仓库覆盖服务器训练源码。在服务器当前项目目录、成功主实验 Python 环境中执行：

```bash
bash shift_diagnostics/server_audit.bash
```

默认核验历史父目录 `$HOME/tievnas2/YinCH`、项目 `GENTLE-consistant`。路径不同则按 README 传入真实目录；模型在别处时增加 search root。

输出：`$HOME/tievnas2/YinCH/shift-diagnostics-v1/audits/A-<UTC时间>-<进程号>/`。每次自动生成新的 `resource_audit.json`、`candidates.json`、`models.json`、`protocol.json`、`source_snapshots.json`、`STATUS.md`、实际 `commands.sh`、`audit_completion.json`。

回传这次审计目录的报告，并同步服务器存在而本地缺少的共享/评估源码。接下来确认全部 58 个模型位置、shared task_manifest/normalizer/离线池及真实环境实现；再实现适配器、运行共享 context smoke。

## 待解决项

- 服务器真实源码/训练版本与 checkpoint 路径，原版 GENTLE 和完整 Ours 的身份。
- 七方法、两环境、四种子的 56 个 iteration 499 模型及两环境 GENTLE seed0/200 采集器。
- 每模型实际训练任务/归一化与主实验共享文件哈希的匹配证据；分离 encoder/policy 同 run/iteration 证据。
- 真实离线池原始字段、坐标、任务归属和终止信息；共享文件 schema。
- 原生适配器、真实环境 smoke 与速度测量。当前没有 A 的端到端通过证据，不能启动正式 B。

每完成阶段，在本文件和服务器对应运行目录 STATUS 中记录：实际命令、退出码、输入版本/哈希、输出路径、验证结果、未完成项和下一步。没有实验数据时只写“待测”。
