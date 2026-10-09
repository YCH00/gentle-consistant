# 第 39 次：生成方式对照与虚拟 consistency 衰减

第 38 次 Ant-Dir 的 `38_semantic_rl_only` 末期在线、离线 context 回报均值最高，但在线提升主要来自 seed 3。加入恒定虚拟 consistency 后，cycle loss 降低，末期回报却低于 RL-only。因此本轮先比较生成方式，再独立检验虚拟 consistency 的使用阶段。

## 两组训练

| 实验名 | 配置 | 已完成的对照 | 改变的因素 |
| --- | --- | --- | --- |
| `39_global_rl_only` | `ant-dir-global-rl-only.json` | `38_semantic_rl_only` | 只将 generation mode 从 semantic 改为 global |
| `39_semantic_vcons_decay` | `ant-dir-semantic-path0-virtual-cons-decay.json` | `38_semantic_virtual_cons`、`38_semantic_rl_only` | 虚拟 consistency 的乘数在 150–300 轮线性衰减到 0 |

真实 consistency 系数始终为 0。Global 组的虚拟 consistency 系数也为 0；衰减组的基础虚拟系数保持 `1/7`，乘以独立的 schedule multiplier。衰减组没有重新启用历史外层 `consistency_loss_weight`，它仍为 0。

| 外层训练轮次 | 0–150 | 225 | 300–499 |
| --- | --- | --- | --- |
| 两组虚拟 RL 权重 | 0.5 | 0.25 | 0 |
| `39_global_rl_only` 虚拟 consistency 有效系数 | 0 | 0 | 0 |
| `39_semantic_vcons_decay` 虚拟 consistency 有效系数 | 1/7 | 1/14 | 0 |

真实 RL/context batch 512、`relabel_data_ratio=0.95`、重构训练、网络、数据、预训练加载要求及 500 轮训练保持匹配。两组均生成 4 个虚拟任务，虚拟 RL batch 为 256，只参与 critic 和 actor Q 项，虚拟 BC 与 semantic/reconstruction/cycle 质量加权均关闭。

语义组邻居数为 2、path steps 为 0、独立训练输入 bank 为 0，支持检查参数沿用第 38 次。Global 组保留 `M=2`、`beta=2`：随机混合两次任务抽样，允许重复端点及负系数外插。它与语义组的输入选取及范围不同，因此这次比较的是两套生成方法的整体表现，不能将差值单独归因于语义距离。若后续需要区分范围与语义度量，再做独立的受限插值对照。

第 300 轮后仍生成任务、计算原始 consistency 诊断，保留原有实验流程；实际虚拟 RL 与虚拟 consistency 梯度在本轮两组都已关闭。关闭 consistency 不关闭真实 context relabel 与重构训练。这两套损失系数衰减与优化器 weight decay 是不同机制。

## 训练命令

上传本地修改后，在服务器项目根目录执行。每条默认训练 seed 0、1、2、3，GPU 0；对照使用已完成的第 38 次结果。

```bash
python train_gentle.py ./configs/interpolation-diagnostics/ant-dir-global-rl-only.json --exp_name 39_global_rl_only --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/ant-dir-semantic-path0-virtual-cons-decay.json --exp_name 39_semantic_vcons_decay --gpu 0
```

本次新增配置限于已有完整分析依据的 Ant-Dir，保持历史配置可复现。

## 一行完成已有三组模型的共享 context 复评

```bash
python evaluate_fixed_context_suite.py --gpu 0
```

该命令依次比较 `37_real_no_cons`、`38_semantic_rl_only`、`38_semantic_virtual_cons` 的 seed 0–3，共 12 个运行，使用每个运行的 200、400、499 轮 encoder/actor/decoder/Q 模型。

每个训练 seed 都以该 seed 的 `37_real_no_cons` 为共同采集器，固定其 200、499 轮 actor 在 z=0 下采集的 context，并加入固定离线 context。每种来源、每个测试任务抽取 3 份 200 条 transition 的 context；同 seed 的三个实验共享同一个 context bank 和相同的 rollout reset seeds，控制期间 z 不再变化。默认每个运行 810 条评估轨迹，共 9720 条，额外在线 context 只在首次创建 bank 时采集。

脚本开始时检查全部运行的 variant、训练 seed、context 协议兼容性及所需模型文件，缺少指定 checkpoint 时直接报错。它调用现有 `evaluate_fixed_context.py`，继续执行归一化、数据来源、模型与缓存校验，不替换 epoch 或静默改变协议。复评按运行顺序执行，不构建训练算法。

输出默认位于 `output/39_fixed_context/ant-dir/`：

| 输出 | 内容 |
| --- | --- |
| `seedN/context_bank/` | 一个训练 seed 内所有实验共享的 context/probe 与来源记录 |
| `seedN/<实验名>/fixed_context_results.json` | 完整逐任务/来源/context 重复结果，包括表示、decoder probe 和 rollout Q/MC 诊断 |
| `per_seed_summary.csv` | 每个训练 seed、checkpoint、context 来源的平均回报 |
| `group_summary.csv` | 上一项在训练 seed 间的均值及样本标准差 |
| `paired_summary.csv` | 相同训练 seed 的实验配对差值、样本标准差和改善 seed 数 |
| `suite_manifest.json` | 设置、运行清单、完成进度；仅 `status=complete` 表示本次请求全部完成 |

先在各 context 重复中平均 reset 回报，再等权平均任务/context，最后汇总训练 seed。评估 reset 或 context 重复不算新增训练 seed。只复评一个训练 seed 时，训练 seed 标准差留空。复评过程中遇到错误，manifest 记录 failed，不发布本次部分结果的汇总；旧汇总文件可能仍在，解读前检查 manifest 的状态和设置。

固定 context 的绝对回报与原始 `_expl` 采用不同协议，应在这套复评内比较实验。相同 context 可排除采集分布变化，仍需结合表示、decoder 和 Q/MC 诊断区分 encoder 与控制能力，不能仅凭回报确认真实任务流形。详细单运行协议见 [固定 context 复评](../../docs/fixed_context_diagnostics.md)。

查看完整计划和文件是否齐全：

```bash
python evaluate_fixed_context_suite.py --dry-run
```

只先检查波动较明显的 seed 1、3：

```bash
python evaluate_fixed_context_suite.py --seed_list 1 3 --output-root ./output/39_fixed_context_focus/ant-dir --gpu 0
```

小规模验证使用独立目录：

```bash
python evaluate_fixed_context_suite.py --seed_list 1 --context-repeats 1 --eval-seeds 0 --output-root ./output/39_fixed_context_smoke/ant-dir --gpu 0
```

完成第 39 次训练后，还可在同一共享协议内加入新模型：

```bash
python evaluate_fixed_context_suite.py --experiments 37_real_no_cons 38_semantic_rl_only 38_semantic_virtual_cons 39_global_rl_only 39_semantic_vcons_decay --gpu 0
```

## 验证与结果判断

本地验证命令：`python -m pytest tests -q`。检查实际 CLI 合并后的参数匹配、两种生成分支的真实网络更新、150/300 前后的有效权重与关闭分支，以及共享 context 复评的端到端运行、缓存复用、异常停止和按训练 seed 汇总。小环境验证不代替服务器实际 MuJoCo 运行及多 seed 性能实验。

训练结果继续对相同未平滑窗口先计算每个 seed 均值，再比较配对差值和波动。Global 对照用于判断语义生成的净贡献；衰减组同时与恒定虚拟 consistency 和 RL-only 比较，以判断前中期约束是否有保留价值。两组都是待验证的实验，不预设它们一定提高回报。
