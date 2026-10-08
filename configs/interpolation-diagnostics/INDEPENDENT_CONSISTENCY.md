# 第 38 次：独立控制真实与虚拟 consistency

第 37 次 Ant-Dir 关闭真实 consistency 后，末 100 轮在线测试回报均值提高，但离线 context 回报降低；仅使用配对 replay 输入没有恢复稳定优势，且一个 seed 出现 critic 明显增长。这轮先降低真实 consistency 权重，并把语义虚拟 RL 与虚拟 consistency 分开。0.1 是待验证的候选系数，不是已经确定的最优值。

## 三组配置

| 实验名称 | 配置 | 比较对象与研究因素 |
| --- | --- | --- |
| `38_real_cons01` | `ant-dir-real-low-cons.json` | 只把第 36 次无虚拟任务对照的 `consistency_loss_weight` 从 `0.35714285714285715` 降为 `0.1`；与已完成的第 36 次及 `37_real_no_cons` 比较 |
| `38_semantic_rl_only` | `ant-dir-semantic-path0-rl-only.json` | 从 `37_real_no_cons` 出发，加入第 34 次 path0 语义任务及虚拟 RL；真实、虚拟 consistency 均不进入 encoder 梯度 |
| `38_semantic_virtual_cons` | `ant-dir-semantic-path0-virtual-cons.json` | 只在上一组上加入虚拟 consistency，系数为 `0.14285714285714285`；真实 consistency 仍关闭 |

三组保持真实 RL/context batch 为 512，`relabel_data_ratio=0.95`、重构权重、输入模式 `legacy`、网络、数据、预训练加载要求及 500 轮训练设置不变。低权重组的 JSON 与第 36 次只差一个字段，两组语义配置互相只差虚拟 consistency 系数。关闭 consistency 不关闭原有 context relabel 与重构训练。

语义两组都保持 `n_vt=4`、语义邻居数 2、`virtual_semantic_path_steps=0`、训练输入池大小 0，以及第 34 次的支持检查参数。虚拟 RL 初始权重 0.5，在第 150–300 轮线性衰减到 0，只参与 critic 与 actor Q 项，不参与虚拟 BC。语义/reconstruction/cycle 质量权重继续关闭，避免同时改变加权机制。

虚拟 consistency 使用 constant schedule，因此在第 300 轮虚拟 RL 关闭后，该组仍保留虚拟 consistency。此选择沿用第 34 次的 constant consistency 设定，能够比较它的持续作用；不是与 RL 联动关闭。另两组的原始 consistency loss 也仍计算，仅作为诊断。Cheetah-Vel 缺少对应的完整结果，本轮新增的训练配置限于已有分析依据的 Ant-Dir。

## 权重含义与兼容性

新增 `real_consistency_loss_weight`、`virtual_consistency_loss_weight`，默认均为 `None`（JSON 为 `null`），表示沿用历史归一化。令每个真实/虚拟任务的原始 consistency loss 为 \(\ell_i\)，实际参与本次训练的数量为 \(N_r,N_v\)：

\[
L_{\rm encoder}=w_{\rm recon}L_{\rm recon}
+a_r\frac{1}{N_r}\sum_{i=1}^{N_r}\ell_i
+a_v(t)\frac{1}{N_v}\sum_{j=1}^{N_v}\ell_j.
\]

当未指定独立系数时，\(a_r=w_cN_r/(N_r+N_v)\)，\(a_v(t)=w_c\gamma(t)N_v/(N_r+N_v)\)。显式指定真实系数时直接使用它；显式指定虚拟系数时直接使用它乘以原有 \(\gamma(t)\)。不会再乘 `consistency_loss_weight` 或任务占比。只指定一类时，另一类继续使用历史规则。

因此 `consistency_loss_weight=0` 时，显式的非零虚拟系数仍能启用虚拟 consistency。第 38 次虚拟系数 `0.5*4/14` 对齐第 34 次生成 4 个虚拟任务时的名义有效系数；独立系数不会因虚拟任务数变化再次归一化。没有生成虚拟任务时，其实际系数为 0，真实系数照常计算。生成失败时也可能从虚拟 replay 中采到此前的数据；应同时查看生成数量与 RL batch，二者含义不同。

两项均为 `None` 时保留历史的合并采样、loss 归约和更新顺序。独立系数启用后，真实与虚拟 consistency 分开计算；某一项系数为 0 时，该项不进入反向传播，防止 `0*NaN` 或其零梯度路径污染另一项。原始 loss 仍会出现在日志中，判断是否关闭要看实际系数。非语义模式改为分开采样时，后续随机数流可能变化。

这两个参数也支持同名 CLI 覆盖，系数必须有限且非负。它们控制任务 consistency，与优化器的 weight decay 无关。

## 整轮诊断

新增诊断沿训练中已有张量计算，不额外调用模型、不改变优化器梯度、不消耗随机数。CSV 与 TensorBoard 都经 `eval_statistics` 发布，不只留在 `self.loss`。

| 日志 | 含义 |
| --- | --- |
| `real_consistency_effective_weight_itr_mean`、`virtual_consistency_effective_weight_itr_mean` | 真实、虚拟任务均值前的实际系数 |
| `consistency_encoder_loss_itr_mean` | 实际加入 encoder 的加权 consistency 项；两项关闭时严格为 0 |
| `real_q1_pred_*`、`real_q2_pred_*` | critic 更新前，在真实 replay 行为动作上的 Q 分布 |
| `real_q_target_*` | 已包含 reward scale、终止标记与 discount 的 Bellman 目标 |
| `real_q_bootstrap_*` | Bellman 更新使用的 target Q1/Q2 最小值，尚未乘 discount 与终止标记 |
| `virtual_q1_pred_*`、`virtual_q2_pred_*`、`virtual_q_target_*`、`virtual_q_bootstrap_*` | 虚拟 replay 对应的分布；无虚拟 RL 行时为 0，结合 `virtual_transition_batch_size` 解读 |
| `real_td_abs_mean/max`、`virtual_td_abs_mean/max` | 两个 critic 相对 Bellman 目标的绝对残差 |
| `qf1_grad_norm_itr_mean/max`、`qf2_grad_norm_itr_mean/max` | backward 后、optimizer step 前的全参数梯度 L2 范数 |
| `policy_q_abs_mean_itr_mean`、`policy_q_lambda_itr_mean` | actor 使用的加权平均 \(|Q|\) 以及原公式 `bc_weight/max(mean_abs_Q,1e-6)`；Q 取自 critic 更新后 |

分布的 `*` 为 `mean/std/min/max`。`_itr_mean` 对整轮每次训练调用等权平均；例如 std 的整轮均值是各 batch 的 population std 均值，不是所有样本合并后的标准差。`*_max_itr_max` 和 `*_min_itr_min` 分别保留整轮样本极值，Q loss、梯度范数及 lambda 另有 `_itr_max`，lambda 另有 `_itr_min`。它们是变化训练 batch 上的统计，不能替代固定 probe 的 Q 校准诊断。

旧的 `Q Predictions`、`QF Loss` 仍保持首个 batch 的统计口径；新增 `qf_loss_itr_mean`、`q_target_itr_mean` 等用于整轮观察。诊断不会修复 critic 发散，本轮没有同时修改 critic 损失、裁剪阈值或学习率。是否需要独立的稳定性实验，结合这些日志与 [固定 context 复评](../../docs/fixed_context_diagnostics.md) 再判断。

## 运行命令

同步本地代码后，在服务器项目根目录依次运行；每条默认 seed 0、1、2、3，GPU 0。已有第 36、37 次对照可直接比较。

```bash
python train_gentle.py ./configs/interpolation-diagnostics/ant-dir-real-low-cons.json --exp_name 38_real_cons01 --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/ant-dir-semantic-path0-rl-only.json --exp_name 38_semantic_rl_only --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/ant-dir-semantic-path0-virtual-cons.json --exp_name 38_semantic_virtual_cons --gpu 0
```

优先运行低权重组；后两组用于重新判断语义任务在真实 consistency 关闭后是否仍有帮助。比较相同未平滑评估窗口的每个 seed，并同时查看在线/离线 context 回报。低权重组需要检验两者的取舍，虚拟 RL 组需要与 `37_real_no_cons` 比较，虚拟 consistency 组需要与 `38_semantic_rl_only` 比较。

本地检查命令为 `python -m pytest tests -q`。测试覆盖独立系数与梯度、关闭项的非有限诊断隔离、旧配置更新和随机数兼容性、critic 目标与分布、整轮累计/reset、CLI 和训练边界。CPU 检查不代替完整 MuJoCo 训练和多 seed 性能验证。
