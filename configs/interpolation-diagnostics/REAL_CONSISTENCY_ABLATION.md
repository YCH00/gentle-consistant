# 第 37 次：真实任务 consistency 的独立消融

本轮只新增 Ant-Dir 的两组实验，与已完成的 `36_real_only_matched` 比较。第 36 次 Ant-Dir 在离线 context 与在线 context 下存在明显回报差距，这两组用于检查真实任务 consistency 的作用及其输入构造；该差距本身不能证明 consistency 是原因。Cheetah-Vel 尚缺少完整的第 36 次对照，本轮不新增其第 37 次配置。

两份配置逐字段继承 `ant-dir-real-only-matched.json`，分别只改变一个因素；第 34、35、36 次历史配置保持原样。

| 实验 | 配置 | 相对第 36 次的唯一改动 |
| --- | --- | --- |
| `36_real_only_matched` | `ant-dir-real-only-matched.json` | 已有对照，consistency 权重为 `0.35714285714285715`，沿用旧输入方式 |
| `37_real_no_cons` | `ant-dir-real-no-cons.json` | `consistency_loss_weight=0.0` |
| `37_real_paired_cons` | `ant-dir-real-paired-cons.json` | `real_consistency_input_mode=paired_replay`，consistency 权重保持 `0.35714285714285715` |

三组均保持 `n_vt=0`，虚拟 transition 初始和最终权重为 0，真实 RL 和 context batch 为 512，`relabel_data_ratio=0.95`，重构损失与其权重、网络、数据、预训练模型加载要求以及训练轮数不变。context batch 中原有 relabel 流程仍提供 486 条，直接采样提供 26 条。关闭 consistency 不等于移除这套 relabel 流程，也不等于关闭 encoder 的重构训练。

## 关闭真实 consistency

`37_real_no_cons` 仍按旧流程采样并计算原始 consistency loss，以保留这部分采样路径和诊断；权重为 0 时 encoder 只对重构损失反向传播，不将 consistency 分支接入总损失，避免 `0 * NaN` 污染更新。原始 loss 日志可能非零，应以 `real_consistency_effective_weight` 为 0 判断是否关闭。训练更新改变后，相同 seed 不保证之后的模型输出相同。

## 改为配对 replay 输入

旧方式默认保留为 `real_consistency_input_mode=legacy`。Ant-Dir 对照中 `consistency_use_policy_relabel_data=false` 并不代表配对 replay：旧 consistency 输入会从锚任务采样状态，再从其他任务采样动作，组合为 `(s,a)`。

`paired_replay` 只改变真实任务 consistency 的输入来源：每个真实任务使用自身 replay 中同一条 transition 的状态和动作，保持 `(s,a)` 行配对关系。decoder 仍以这组输入和原任务 z 生成奖励；如启用动力学预测，仍由 decoder 生成后继状态。随后 encoder 从生成的 context 恢复目标 z，cycle 目标形式保持不变。没有用 replay 中真实奖励替代 decoder 奖励，否则会同时改变待研究的目标。

保持行配对能确保输入来自已记录的任务数据，但不保证 decoder 预测准确，也不保证改善策略回报。此模式针对真实任务项，不改变虚拟任务的生成方式。本轮 `n_vt=0`，不会生成虚拟任务。

该选项也可通过 `--real_consistency_input_mode` 覆盖，支持 `legacy`、`paired_replay`、`policy`、`cross_task`。显式模式只覆盖真实任务输入；`legacy` 沿用原布尔选项，保留历史调用顺序。配对模式减少了其他任务动作的抽样，因此与旧输入模式不保证后续随机数流一致。

## 运行命令

同步最新本地代码到服务器后，在项目根目录运行。每条命令默认执行 seed 0、1、2、3；以下均使用 GPU 0，应依次运行。无需重跑完整的 Ant-Dir 第 36 次对照。

```bash
python train_gentle.py ./configs/interpolation-diagnostics/ant-dir-real-no-cons.json --exp_name 37_real_no_cons --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/ant-dir-real-paired-cons.json --exp_name 37_real_paired_cons --gpu 0
```

## 检查与分析

首先核对保存的配置、预训练模型路径及 SHA256 与对应 seed 的第 36 次记录一致，并确认日志和最终模型齐全。两组的虚拟任务数、虚拟 transition 加入量及虚拟 RL batch 数都应为 0；真实 consistency 有效权重应分别为 0 和 `0.35714285714285715`。

日志新增 `real_consistency_paired_replay` 和 `real_consistency_paired_replay_itr_mean`：配对组为 1，无 consistency 组为 0。它们只表示输入模式是否启用，不表示 decoder 预测质量。

比较相同评估区间内每个 seed 的平均回报，再汇总配对差值及 seed 间波动。在线和离线 context 回报应同时查看，并结合 [固定 context 复评](../../docs/fixed_context_diagnostics.md) 排查 context 来源、采集策略与模型阶段的影响。两组都改善时，配对输入构造值得优先验证；仅关闭 consistency 改善时，应继续研究目标及权重；均无改善时，再单独检查 batch、relabel 等共同机制。四个 seed 的均值差异仍不等于稳定收益或因果机制已经确认。

本地验证不代替 MuJoCo 完整训练，本轮改动的完整 MuJoCo 性能尚未验证。
