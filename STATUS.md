# STATUS — shift-diagnostics-v2

更新：2026-11-02（Asia/Shanghai）。当前项目：`gentle-consistant`。

**当前：阶段 A 进行中。服务器资源审计已回传，7 个报告文件 SHA256 全部通过；源码/配置正文待轻量导出，模型适配和真实环境 smoke 未完成。B/C/D 尚未开始。** 用户负责同步新增文件并在服务器运行。

2026-11-02 用户明确指定：`23_local_decay` 和 `28_global_decay` 分别作为两个完整 Ours 条目，`0_gentle_base` 作为 GENTLE。协议因此升级为 v2；原 v1 审计保留不变，只作为资源发现证据。ER-TRL 的 cheetah-vel 正在训练（用户确认）。

## 固定范围

- 比较 FOCAL、CORRO、CSRO、GENTLE、UNICORN-SS、ER-TRL、Ours-local、Ours-global 八个完整方法条目，不做组件消融。
- 固定目录映射：GENTLE=`0_gentle_base`，Ours-local=`23_local_decay`，Ours-global=`28_global_decay`；两个 Ours 分别报告，使用同一份共享 context。
- 环境：cheetah-vel、walker-rand-params；模型 seeds 0/1/2/3、iteration 499。
- 原版 GENTLE seed0 的 iteration 200/499 为唯一共享采集器；所有方法读取同一份冻结 raw context bank。
- H=200、K=20、L=3；每 rollout 最多 200 步，固定 context，不追加反馈，不沿用累计 600 步的主实验评估。
- 当前规模：64 个评估模型 + 2 个额外 iteration 200 采集器 = 66 个模型位置；正式评估共 384,000 episodes。除方法条目扩展外，原交接文档的采集、冻结、统计和防泄漏规则保持适用。
- 交接文档：`docs/SHIFT_ROBUSTNESS_EXPERIMENT_HANDOFF.md`，与用户提供文件 SHA256 相同：`4b16fa5dc8ffdd1daa7b0a8feb56e38040f22c40039f808bfef09766ebf8a850`。

## 阶段状态与下一步

| 阶段 | 当前状态 | 进入下一阶段的条件 |
| --- | --- | --- |
| A：核验与适配 | IN_PROGRESS：服务器审计已验证；轻量证据导出待执行 | 完成资源绑定与原生适配；每方法/环境使用同一 raw context 的真实 smoke 通过 |
| B：bank 与实验 1 | NOT_STARTED | 每环境 2,000 contexts；每方法/seed/环境 6,000 episodes，四种子完整；回报/latent/哈希可核验 |
| C：实验 2 | NOT_STARTED | 独立 probe 数据无轨迹交叉；seen-only 训练/验证；同一 query 跨方法共享 |
| D：实验 3 与论文材料 | NOT_STARTED | 保存 latent/投影配置可重画；逐样本结果可重算表格 |

## 初次本地核验记录（历史）

- 初次本地 Git HEAD：`6e44eecb5427e57a8ac3453858d53f3f418f6908`。任务开始前 `.gitignore` 已修改，当时没有改动该文件或训练代码。
- 当前目录及已检查父目录未发现适用的 AGENTS.md。
- 有 `train_gentle.py`、Agent/actor/环境实现和两个环境配置；静态接口证据在 `shift_diagnostics/ADAPTER_NOTES.md`。
- 缺少 `evaluate_fixed_context.py`、`evaluate_fixed_context_suite.py`；`shared_protocol` 当前仅缓存、无 Python 源码。
- 排除旧测试产物后，本次当前仓库扫描未发现 checkpoint 候选或真实共享 task_manifest/normalizer。扫描范围、深度和排除项保存在 resource_audit；这不等于证明服务器缺失。
- 本地相邻目录存在其他 baseline 源码入口和多个 GENTLE 版本，但尚未验证对应正式 run；不能将目录名视为模型身份。
- 当前本地 Python 为 3.12.3；包元数据未找到 numpy/torch/gym/mujoco-py/sklearn。本轮没有运行真实环境或加载模型。
- 本地环境配置中的历史 data_dir 不一致，未擅自修改；应以服务器成功主运行的共享文件和配置为准。

## 本轮新增文件

- `shift_diagnostics/audit.py`：只读资源发现；模型矩阵现按 v2 生成 66 个位置，不会自动把候选提升为正式模型。
- `shift_diagnostics/protocol.json`：v2 固定协议；deviations 记录用户指定两个 Ours 的变更，共享资源绑定待核验。
- `shift_diagnostics/server_audit.bash`：服务器一条命令生成独立审计目录。
- `shift_diagnostics/README.md`：A→B→C→D 顺序、当前可执行命令、每阶段记录模板。
- `shift_diagnostics/ADAPTER_NOTES.md`：已有源码接口和适配注意点。
- `shift_diagnostics/test_audit.py`：审计护栏检查，不是实验 smoke。
- `shift_diagnostics/review_audit.py`：验证回传报告、生成候选模型草稿及证据导出清单。
- `shift_diagnostics/export_evidence.py` 和 `evidence_request.json`：读取已定位源码、训练配置及小型共享文件，导出 ZIP；不加载、复制或重新哈希权重。
- `shift_diagnostics/test_evidence.py`：证据包哈希、变更提示、不覆盖、不包含权重及路径边界检查。

## 实际运行记录

工作目录均为 `D:/Desktop/offline meta RL/gentle-consistant`。

1. 本地源码与资源读取：读取交接文档、训练配置、Agent、actor、normalizer、环境包装和注册；`git -c safe.directory='D:/Desktop/offline meta RL/gentle-consistant' rev-parse HEAD` 核验版本。Git 信任只在单条命令生效，未修改全局配置。
2. `python -m unittest discover -s shift_diagnostics -p test_audit.py -v`：第一次退出码 0，3 项通过（完整模型矩阵、协议与交接文档哈希、候选/排除/不反序列化/不覆盖/报告哈希）。后续最终复验结果追加于下方。
3. `python shift_diagnostics/audit.py --repo-root . --hash-checkpoints --out output/shift-diagnostics-v1/audits/local-A-20261102-01`：退出码 1；本地旧 output 目录 WinError 5 写权限拒绝，没有完成报告。未更改 ACL，改用项目中新目录。
4. `python shift_diagnostics/audit.py --repo-root . --hash-checkpoints --out diagnostics_audits/local-A-20261102-01`：退出码 0，资源发现完成；checkpoint 候选 0，shared Python 源码 0，scan_errors 为空。报告中的 A_complete=false。此报告保留其生成时的源码哈希；最终工具修订后使用新审计目录复核。
5. 最终复验 `python -m unittest discover -s shift_diagnostics -p test_audit.py -v`：退出码 0，3/3 通过。所有新增 Python 文件通过 `ast.parse(..., feature_version=(3,7))` 语法检查；这不是 Python 3.7 运行时实测。`C:/Program Files/Git/bin/bash.exe -n shift_diagnostics/server_audit.bash` 退出码 0。
6. `python shift_diagnostics/audit.py --repo-root . --hash-checkpoints --out diagnostics_audits/local-A-20261102-02`：退出码 0。**当前有效本地审计目录为此路径**；checkpoint 候选 0、shared Python 源码 0、scan_errors 为空；两个候选仅为 train_gentle.py 和本轮拟定 protocol.json，不是正式共享数据。
7. 使用 Python hashlib 对 `local-A-20261102-02/audit_completion.json` 中 7 个输出文件逐个复核 SHA256，全部通过；交接文档 SHA256 与协议匹配，audit.py 当前源码与报告一致。A 的真实模型/环境 smoke 仍未执行。

## 服务器审计回传与本轮复核

- 服务器实际执行：`/root/miniconda3/envs/gentle/bin/python /root/tievnas2/YinCH/GENTLE-consistant/shift_diagnostics/audit.py --repo-root /root/tievnas2/YinCH/GENTLE-consistant --search-root /root/tievnas2/YinCH --search-root /root/tievnas2/YinCH/shared_protocol/shared-main-v1 --hash-checkpoints --out /root/tievnas2/YinCH/shift-diagnostics-v1/audits/A-20261102T110236Z-3639`。
- 用户确认执行完成；报告带完成标记，原始退出码没有单独记录。报告生成时间为 2026-11-02 19:44:26 +08:00，scan_errors 为空。
- 本地回传路径：`shift-diagnostics-v1/audits/A-20261102T110236Z-3639/`；7 个输出文件 SHA256 均通过，不改写原始报告。
- 服务器环境：Python 3.8.20、numpy 1.23.5、torch 2.4.1、gym 0.17.2、mujoco-py 1.50.1.68、sklearn 1.3.2；包版本不能替代环境 smoke。
- 服务器及当前本地 Git HEAD 都是 `49857cc2bc0f7d343b1695c7e645497152982459`。当前本地 hopper/walker 配置已有用户修改，本轮保留。train_gentle.py、gentle.py、rl_algorithm.py、agent.py 的文件字节受 CRLF/LF 影响；转为 LF 后与服务器审计源码哈希一致。
- 两个目标环境均找到共享 task_manifest、observation_normalizer、dataset_manifest 和 preparation_provenance 的路径与哈希；文件内容待导出。`0_gentle_base` 两环境 seed0 的 200/499 权重存在；三个用户选定目录均有四种子 499 候选，配置/任务/归一化契约仍未验证。
- 定位到 FOCAL-v2、CORRO-native-v2、backup/CSRO、UNICORN-ss-v2、ER-TRL-v2 等资源。last.pt/final.pt 的 iteration 不能靠名字推断。
- ER-TRL cheetah-vel 仅发现 smoke.pt；用户确认正式训练中。不得用 smoke 补位，其他准备继续。
- 服务器交接文档缺失，应在下一次同步补上 `docs/SHIFT_ROBUSTNESS_EXPERIMENT_HANDOFF.md`。
- 已执行 `python shift_diagnostics/review_audit.py --audit-dir shift-diagnostics-v1/audits/A-20261102T110236Z-3639 --out diagnostics_audits/server-A-review-v2-20261102-01`，退出码 0。输出 review.json、models_unverified.json、protocol.json、evidence_request.json 和 REVIEW.md；导出请求已复制到 shift_diagnostics/。
- 证据请求仅选择用户指定的三个 GENTLE/Ours run 目录，另加 baseline 资源，共 230 个元数据文件和 7 个源码根目录。它是配置/源码核验，不是消融评估。
- `python -m unittest discover -s shift_diagnostics -p 'test_*.py' -v`：退出码 0，5/5 通过。证据导出测试使用临时 fixture，仅验证文件处理，不构成真实环境 smoke。
- 最终配置检查：66 个模型位置中 26 个用户指定位置已绑定候选文件（24 个 GENTLE/Ours 评估模型 + 2 个采集器），4 个 ER-TRL Cheetah 位置 pending_training，其余 36 个位置尚未完成绑定；`missing` 在此仅指缺少已验证绑定。导出请求与当前 v2 协议完全一致；64 × 6,000 = 384,000 episodes 校验通过。
- 新增 Python 文件均通过 Python 3.7 语法解析，Bash 脚本通过 `bash -n`，本轮文件 `git diff --check` 通过。没有运行实际模型加载、环境 smoke 或正式实验。

## 服务器下一条命令（计划，尚未执行）

同步更新后的 `shift_diagnostics/`、本 STATUS 和原始交接文档。在服务器执行下面的轻量证据导出；无需重跑全盘审计：

```bash
cd /root/tievnas2/YinCH/GENTLE-consistant
/root/miniconda3/envs/gentle/bin/python shift_diagnostics/export_evidence.py \
  --audit-dir /root/tievnas2/YinCH/shift-diagnostics-v1/audits/A-20261102T110236Z-3639
```

输出：`/root/tievnas2/YinCH/shift-diagnostics-v2/evidence/A-evidence-<时间>-<进程号>/`。下载这个新目录（含 evidence.zip、bundle_manifest.json、completion.json、STATUS.md、commands.sh）到本地即可；不需要解压或下载模型。若有错误，保留目录与终端输出供核验。

下一步读取配置和源码正文，验证任务/归一化/iteration 与权重配对，完成原生适配和共享 context smoke；ER-TRL Cheetah 完成训练后再补其正式模型验收。

## 待解决项

- 用户已指定 GENTLE/Ours run 目录；仍需训练配置、实际加载实现和预处理契约的正文证据。
- 八个条目、两环境、四种子的 64 个 iteration 499 模型及两环境 GENTLE seed0/200 采集器；ER-TRL Cheetah 训练中。
- 每模型实际训练任务/归一化与主实验共享文件哈希的匹配证据；分离 encoder/policy 同 run/iteration 证据。
- 真实离线池原始字段、坐标、任务归属和终止信息；共享文件 schema。
- 原生适配器、真实环境 smoke 与速度测量。当前没有 A 的端到端通过证据，不能启动正式 B。

每完成阶段，在本文件和服务器对应运行目录 STATUS 中记录：实际命令、退出码、输入版本/哈希、输出路径、验证结果、未完成项和下一步。没有实验数据时只写“待测”。
