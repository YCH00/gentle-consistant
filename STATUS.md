# STATUS — shift-diagnostics-v3

更新：2026-11-03（Asia/Shanghai）。服务器：`/root/tievnas2/YinCH/GENTLE-consistant`。

**A 进行中：已校验证据包并实现原生适配和真实环境 smoke 入口，等待服务器运行。B/C/D 未开始。**

## 已确认的协议

- 八个完整条目：FOCAL、CORRO、CSRO、GENTLE、UNICORN-SS、ER-TRL、Ours-local、Ours-global，不做消融。
- GENTLE=`0_gentle_base`；Ours-local=`23_local_decay`；Ours-global=`28_global_decay`。分别报告两个 Ours，不按回报挑选。
- 用户批准复用各方法主实验原生预处理及已验证的最终模型，同时严格共享同一 raw context。新协议/输出升级为 v3，保留 v1/v2 证据。
- 环境：cheetah-vel、walker-rand-params；训练 seeds 0/1/2/3。
- 最终 iteration（0-based）：FOCAL/CSRO/GENTLE/Ours=499；CORRO/UNICORN-SS=999；ER-TRL=99。
- FOCAL/CORRO/UNICORN-SS/ER-TRL 使用原生 raw observation；CSRO 使用 shared RMS；GENTLE/Ours 使用原生训练 RMS。共享 bank 保持 raw obs/s'、normalized action、raw reward。
- 唯一共享采集器：GENTLE seed0、iteration200/499，z=0、确定性 actor。
- 正式 H=200、K=20、L=3：每环境 2,000 contexts，每模型/seed/环境 6,000 episodes，总计 384,000 episodes。
- 原交接文档 `docs/SHIFT_ROBUSTNESS_EXPERIMENT_HANDOFF.md` 保留，SHA256=`4b16fa5dc8ffdd1daa7b0a8feb56e38040f22c40039f808bfef09766ebf8a850`。用户上述决定覆盖旧的单 Ours、统一 iteration/预处理约定；其余冻结、统计、防泄漏规则继续适用。

## 阶段状态

| 阶段 | 当前状态 | 下一步 |
| --- | --- | --- |
| A：核验与适配 | IN_PROGRESS | 服务器运行下方入口，回传全部输出，验收真实模型与 MuJoCo |
| B：共享 bank、实验 1 | NOT_STARTED | A 通过后实现/运行正式采集和固定 context 评估 |
| C：实验 2 | NOT_STARTED | 复用 B 回报；独立 probe 数据、seen-only 训练/验证 |
| D：实验 3、论文材料 | NOT_STARTED | 复用 B latent；可重画图、可重算表格 |

## 已执行的资源核验

v1 审计目录：`shift-diagnostics-v1/audits/A-20261102T110236Z-3639/`。7 个报告成员哈希通过，不需重跑全盘审计。

用户已实际执行：

```bash
cd /root/tievnas2/YinCH/GENTLE-consistant
/root/miniconda3/envs/gentle/bin/python shift_diagnostics/export_evidence.py \
  --audit-dir /root/tievnas2/YinCH/shift-diagnostics-v1/audits/A-20261102T110236Z-3639
```

输出：`/root/tievnas2/YinCH/shift-diagnostics-v2/evidence/A-evidence-20261102T121715Z-5124/`。Files=2203、errors=0、changed_metadata=0；ZIP 和全部成员校验通过，ZIP SHA256=`8d332a9e5910243e5e6a40cb7127961df5f6129c8f08cbd554cd43e78650568c`。导出成功不代表 A 完成。

本地解包：`diagnostics_audits/evidence-expanded-20261102/files/`。已据此生成 `shift_diagnostics/stage_a_plan.json`，固定服务器路径、源码和共享文件哈希。

历史 STATUS 和初次接口笔记保存在 `diagnostics_audits/STATUS-v2-before-native-approval-20261102.md`、`ADAPTER_NOTES-initial-20261102.md`。

## 本地实现与验证

- 已实现：输入核验、最终模型选择、六套原生推理接口（覆盖八个条目）、真实环境任务恢复、独立进程 smoke、命令/结果/状态记录。
- 只构建推理网络，严格加载原生 encoder/actor/CORRO attention；不构建训练循环。UNICORN-SS 表征计算 E[tanh(Z)]，控制仍使用原生随机 latent。
- 每环境先生成唯一 smoke bank：1 个 seen task × 2 contexts × 200 行。所有模型共享该 bank 和配对 reset seeds；检查有限奖励、动作、表征确定性、权重和统计不变。
- 校验原始离线池全部文件的哈希、任务、形状、动作坐标。GENTLE/Ours 的 RMS 仅从已验证的原始 seen 训练文件重建；记录顺序、哈希和与 shared RMS 的差异，不在 unseen/context 上拟合。
- 本地实际命令：`python -m unittest discover -s shift_diagnostics -p 'test_*.py' -v`，2026-11-03 最终复核 13 项全部通过（退出码 0）。原生接口测试包含 6 套实现 × 2 环境的 12 个子进程案例，使用随机测试权重，不计入 A；新增检查证明 unseen 文件不会进入训练 RMS。日志与验证回执：`diagnostics_audits/local-validation-v3-20261103/tests.log`、`validation.json`。
- 本地测试依赖：Python3.12 / NumPy1.26.4 / PyTorch2.5.1 CPU，来自 `output/test-runtime/py312/Lib/site-packages`。全部诊断 Python 文件已通过 Python3.8 语法解析；入口 help、Bash 语法与 git diff --check 通过。
- 尚未在服务器运行本轮 smoke；无真实模型 smoke 通过记录，无正式实验结果。

## 用户现在执行

1. 上传整个 `shift_diagnostics/`（包括 `stage_a_plan.json`、`protocol.json`）及根 `STATUS.md` 到服务器项目对应位置。无需上传 `diagnostics_audits/` 或已下载的 v1/v2 目录。保留原有服务器训练源码。
2. 执行：

```bash
cd /root/tievnas2/YinCH/GENTLE-consistant
/root/miniconda3/envs/gentle/bin/python shift_diagnostics/run_stage_a.py --device cpu
```

3. 下载打印的整个目录：`/root/tievnas2/YinCH/shift-diagnostics-v3/smoke/A-<UTC时间>-<PID>/`。

默认 CPU 单线程、全部四种子串行检查，避免占用 ER-TRL 训练 GPU。开头需要对大 checkpoint 计算哈希；FOCAL 文件含 replay，加载会使用对应 CPU 内存，但不会恢复训练算法。

每项打印 RUN/PASS/FAIL。输出含 `models.json`、`jobs.json`、`commands.sh`、`completion.json`、`STATUS.md`、输入/源码哈希、共享 smoke bank、逐模型 latent/episode 及日志。退出码 0=A 通过；2=资源未齐或 smoke 失败；1=协调器/输入核验失败。失败也回传整个目录，不自动替换环境或安装依赖。

若默认位置没有已安装的 MuJoCo，用 `--mujoco150 /实际路径/mjpro150 --mujoco131 /实际路径/mjpro131` 指定已有目录。两个后端在独立进程启动前设置动态库路径。

## 待验收与证据边界

- ER-TRL Cheetah 用户确认仍在训练；运行入口会重新检查正式完整四种子目录。未齐时继续检查其他模型，A 保持未完成，不使用 smoke checkpoint。
- baseline 存在多组时，使用完成文件时间最新的完整四种子组；在任何诊断 rollout 前冻结路径/哈希，不读取回报选择。
- 历史 GENTLE/Ours state_dict 不含 seed/source/RMS 签名；记录用户指定同 run/iteration 配对和外部来源。原训练 glob 枚举顺序未保存，RMS 重建采用当前原生顺序并记录与 shared RMS 差异，不能称为逐 bit 已证明相同的历史统计。
- 服务器需要验证真实 checkpoint、依赖和 MuJoCo reset/step。只有 `completion.json` 的 `stage_A_complete=true`，64/64 评估模型和 2/2 额外采集器通过，才进入 B。
- smoke 数据不能用于 B/C/D 正式图表；B/C/D 程序待 A 接口验收后继续补齐。

服务器每次运行在输出 STATUS 和项目根 STATUS 记录实际命令、路径、退出码、验证结果和下一步。下一轮依据回传结果修复或验收 A，再继续 B→C→D。
