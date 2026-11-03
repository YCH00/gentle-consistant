# shift-diagnostics-v3：阶段 A 入口

v1 审计、v2 证据导出已回传校验。按用户确认，八个完整方法保留主实验原生预处理/最终模型，严格共享同一 raw context，不做消融。进度见根 `STATUS.md`，接口说明见 `ADAPTER_NOTES.md`。

## 现在运行

上传整个 `shift_diagnostics/` 和根 `STATUS.md` 到 `/root/tievnas2/YinCH/GENTLE-consistant` 对应位置，包含已生成的 `stage_a_plan.json`。无需上传 `diagnostics_audits/`、旧结果，也无需再次 audit/export。

```bash
cd /root/tievnas2/YinCH/GENTLE-consistant
/root/miniconda3/envs/gentle/bin/python shift_diagnostics/run_stage_a.py --device cpu
```

下载终端打印的整个 `/root/tievnas2/YinCH/shift-diagnostics-v3/smoke/A-<UTC>-<PID>/`。每次创建新目录，CPU 单线程、方法/环境串行子进程，检查全部四种子。FOCAL checkpoint 含 replay，哈希/加载耗时与内存占用取决于原文件大小。

需要时指定已有后端：`--mujoco150 /实际目录/mjpro150 --mujoco131 /实际目录/mjpro131`。若使用 GPU，`CUDA_VISIBLE_DEVICES=1` 对应进程内 `--device cuda:0`。

## 验证流程

1. 校验从证据包固定的源码、所选 GENTLE/Ours 配置和 shared artifacts。
2. 在 rollout 前选择并哈希最终模型；多个 baseline 组使用完成标记最新的完整四种子组，不按回报选择。
3. 核验全量原始离线池，恢复原生训练 RMS；每环境由 GENTLE seed0/499、z=0、确定性 actor 生成一份两个 context 的 raw smoke bank。
4. 各模型在独立进程严格加载原生 encoder/actor/attention，检查配置、种子、训练计数、源码绑定；共享同一 raw bank。
5. 从 manifest 恢复任务，运行真实旧版 MuJoCo reset/step；检查动作、奖励、表征确定性、冻结参数/统计和配对 reset seeds。
6. 64 个评估模型和两个 iteration200 采集器全部通过才完成 A。ER-TRL Cheetah 未齐时明确记录，不以 smoke 权重替代。

| 方法 | 目录/版本 | 最终 iteration（0-based） | 观测预处理 |
| --- | --- | --- | --- |
| FOCAL | FOCAL-v2 / run01 | 499 | identity |
| CORRO | CORRO-native-v2 / shared-native-v2-3windows，generative | 999 | identity |
| CSRO | backup/CSRO / runs/CSRO | 499 | shared RMS |
| GENTLE | 0_gentle_base | 499 | native training RMS |
| UNICORN-SS | UNICORN-ss-v2 / 完整 SS run | 999（checkpoint iteration=1000） | identity |
| ER-TRL | ER-TRL-v2 / 完整正式 run | 99 | identity |
| Ours-local | 23_local_decay | 499 | native training RMS |
| Ours-global | 28_global_decay | 499 | native training RMS |

原始四列离线数据缺少 absorbing-terminal 标签；保留未知状态，以及原生 loader 的轨迹末行 boundary 语义。所有 bank 保持 raw obs/s'、normalized action、raw reward，各模型内部只做一次原生变换。

## 输出与通过条件

- `completion.json`：A 完成状态、通过数量、未完成资源与失败任务。
- `models.json`：路径、哈希、配置、选择依据和验证状态。
- `jobs.json`、`commands.sh`、`logs/`：实际命令、退出码、耗时、错误。
- `banks/<env>/`：唯一 smoke context、原始池核验、训练统计重建与来源。
- `evaluation/<env>/<method>/seedN/`：逐 episode、latent、验证回执；`collector200/` 保存额外采集器检查。
- `STATUS.md`：阶段状态和下一步；项目根 STATUS 也追加运行摘要。

退出码 0=A 全部通过，2=仍有资源/检查未完成，1=输入或协调器失败。失败也下载完整目录。脚本不自动启动 B/C/D。

历史 GENTLE/Ours 没有内嵌 seed/source/RMS 签名；按用户指定同 run 配对。RMS 从 hash-verified seen 训练文件按原生单次更新重建。历史 glob 顺序未存档，记录当前顺序及与 shared RMS 差异，不宣称逐 bit 已证明相同。

## 后续顺序

| 阶段 | 工作与验收 |
| --- | --- |
| B | 正式共享 bank 每环境 2,000 contexts，每 context H=200；全部模型共 384,000 episodes；保存回报/latent，seen/unseen 分别汇总 |
| C | 复用 B 的 policy499_sigma000 回报；独立 probe train/val context/query，轨迹无交叉，seen-only 训练/调参；H0 为诊断预测参照 |
| D | 复用 B latent；固定 4 seen+4 unseen、两条件、每模型 320 点；输出可重画图和可重算表格 |

B/C/D 程序待 A 实际接口验收后补齐；本轮 smoke 数据不用于正式结果。

## 开发检查

```bash
python -m unittest discover -s shift_diagnostics -p 'test_*.py' -v
```

需要 NumPy/PyTorch；原生接口测试仅在本地下载证据存在时执行，使用随机测试权重，不构成真实模型 smoke。服务器无需运行开发测试。

`build_stage_a_plan.py --evidence-dir <证据目录> --out <新计划路径>` 可从 ZIP 重建计划并验证全部成员。本次已提供计划，服务器不用重复构建。
