# 第 35 次实验：输入池与虚拟 consistency 的独立消融

本组配置从 `configs/interpolation-ablation/` 的第 34 次配置复制，只改变各自待验证因素。历史配置保持原样。优先跑 Ant-Dir、Cheetah-Vel 的四个 seed；其余连续环境提供同样的独立配置。Cheetah-Dir 是离散方向对照，继续使用原有 `cheetah-dir-control.json`，不增加虚拟任务。

| 实验名称 | 文件后缀 | 对照 | 唯一研究因素 |
| --- | --- | --- | --- |
| `35_bank_path0` | `-semantic-path0-bank.json` | `34_semantic_path0` | 扩大并独立刷新受支持的训练输入池 |
| `35_bank_path8` | `-semantic-path8-bank.json` | `34_semantic_path8` | 同上，保留原有 8 步路径优化 |
| `35_global_cons_decay` | `-global-consistency-decay.json` | `34_global_control` | 仅让虚拟 consistency 的乘数逐步归零 |

输入池组保留恒定的虚拟 consistency；consistency 组保留原有 global 生成方式。先比较每组与对应的 34 对照，再比较两个 bank 组；不要将同时改变多个因素的差值归因于路径优化。四个 seed 应全部跑到完整轮次，并使用相同区间比较逐 seed 的均值、差值和波动。

## 受支持的独立训练输入池

原有 fit/check 小探针仍用于原型图、语义距离、支持参考、路径优化和验证。新增的 `virtual_semantic_training_bank_size=1024` 表示每个真实任务独立抽取 1024 条配对 transition；每 `virtual_semantic_training_bank_refresh_interval=25` 个梯度更新刷新一次。图仍每 100 个梯度更新刷新。这里的 25/100 不是外层训练轮次。

每条边只从两端任务的候选 transition 中选输入。候选的 `(s,a)` 使用该图原有 fit 探针的均值、标准差归一化，并且必须同时落在两个端点任务的最近邻支持半径内；最多有 2048 个候选行，支持筛选后可能更少。采样仍然有放回，`s,a,s',done` 使用同一行索引；随后按已有逻辑用虚拟 z 解码奖励，动力学任务还解码后继状态。

刷新输入池不调用 decoder、不改变已缓存的图/路径/质量分数。每次图刷新会强制重建输入池，避免使用过期的支持参考。某条边的新池不足 `virtual_semantic_min_shared_probes` 时，回退到该边已经验证通过的旧 check 输入，并记录回退比例，不删除该边或改变路径选择概率。

额外的数据抽样会消耗随机数，训练也会改变编码器，因此不能保证不同实验后续的图完全相同。“图不变”仅指单次输入池刷新不修改当时缓存的几何结构。1024 是候选行数量，不保证 1024 个独立 transition；底层 replay 可能抽到重复行。支持检查是经验约束，不保证 decoder 在新输入上的预测误差小，也不保证恢复真实任务流形。

默认 `virtual_semantic_training_bank_size=0` 保留旧的 check 输入复用方式，因此旧命令不会自动切换到新实验。原有 `n_vt=4`、语义邻居数（Point 为 3，其他连续环境为 2）、支持半径、路径参数、虚拟 RL 权重与衰减均继承第 34 次配置。

## 仅衰减虚拟 consistency

实际编码器 consistency 项为：

\[
L_{\mathrm{cons}}=w_c\frac{\sum_{i=1}^{N_r}\ell_i+\gamma(t)\sum_{j=1}^{N_v}\ell_j}{N_r+N_v}.
\]

`virtual_consistency_weight_schedule=linear_decay` 将乘数 gamma 从 1 线性衰减到 `virtual_consistency_final_weight=0`。已有 `consistency_loss_weight` 保持不变，分母保留实际生成的真实和虚拟任务总数。例如 10 个真实任务、4 个虚拟任务、外层权重 0.5 时，真实均值的系数始终为 `0.5*10/14`，虚拟均值的系数从 `0.5*4/14` 降为 0。没有生成虚拟任务时保留真实任务均值。

| 环境 | gamma 衰减起止外层轮次 |
| --- | --- |
| Ant-Dir | 150–300 |
| Cheetah-Vel | 100–250 |
| Point-Robot | 50–200 |
| Hopper-Rand-Params | 50–200 |
| Walker-Rand-Params | 50–200 |

本组配置让 gamma 与原有虚拟 RL 衰减窗口一致，但两套参数和计算相互独立。gamma 为 0 后仍生成虚拟任务并计算原始虚拟 consistency，以保留消融中的采样与计算流程；这不是节省计算的模式。该改动也不涉及优化器的 weight decay。

默认 `virtual_consistency_weight_schedule=constant` 表示 gamma 恒为 1，保留原先整体平均的运算。四个新参数均可通过同名命令行选项覆盖：`--virtual_consistency_weight_schedule`、`--virtual_consistency_weight_decay_start_itr`、`--virtual_consistency_weight_decay_end_itr`、`--virtual_consistency_final_weight`。输入池的两个参数也支持同名命令行选项。

## 命令

在项目根目录执行；默认四个 seed 为 0、1、2、3，`--gpu` 按实际空闲设备修改。补跑单个 seed 时加 `--seed_list 2` 等选项。不要并发启动以下所有命令挤在同一张 GPU 上。

```bash
python train_gentle.py ./configs/interpolation-diagnostics/ant-dir-semantic-path0-bank.json --exp_name 35_bank_path0 --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/ant-dir-semantic-path8-bank.json --exp_name 35_bank_path8 --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/ant-dir-global-consistency-decay.json --exp_name 35_global_cons_decay --gpu 0

python train_gentle.py ./configs/interpolation-diagnostics/cheetah-vel-semantic-path0-bank.json --exp_name 35_bank_path0 --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/cheetah-vel-semantic-path8-bank.json --exp_name 35_bank_path8 --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/cheetah-vel-global-consistency-decay.json --exp_name 35_global_cons_decay --gpu 0

python train_gentle.py ./configs/interpolation-diagnostics/point-robot-semantic-path0-bank.json --exp_name 35_bank_path0 --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/point-robot-semantic-path8-bank.json --exp_name 35_bank_path8 --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/point-robot-global-consistency-decay.json --exp_name 35_global_cons_decay --gpu 0

python train_gentle.py ./configs/interpolation-diagnostics/hopper-rand-params-semantic-path0-bank.json --exp_name 35_bank_path0 --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/hopper-rand-params-semantic-path8-bank.json --exp_name 35_bank_path8 --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/hopper-rand-params-global-consistency-decay.json --exp_name 35_global_cons_decay --gpu 0

python train_gentle.py ./configs/interpolation-diagnostics/walker-rand-params-semantic-path0-bank.json --exp_name 35_bank_path0 --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/walker-rand-params-semantic-path8-bank.json --exp_name 35_bank_path8 --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/walker-rand-params-global-consistency-decay.json --exp_name 35_global_cons_decay --gpu 0
```

## 需要同时看的诊断

- `virtual_semantic_training_supported_input_count_mean_itr_mean`：每条边通过支持筛选的候选行数，观察是否突破原先约 64 个 check 输入的上限。
- `virtual_semantic_training_bank_edge_coverage_fraction_itr_mean`：新池足够大、可以使用新池的边占比。
- `virtual_semantic_training_bank_fallback_fraction_itr_mean`：实际生成的虚拟任务中，回退旧 check 输入的比例；高回退比例意味着新池没有充分生效。
- `virtual_semantic_sampled_support_unique_fraction_itr_mean`：每个虚拟 context 中不同池内行索引的比例；不是原始 transition 的去重率。
- `virtual_semantic_training_bank_refresh_count`：训练输入池累计刷新次数。
- `virtual_consistency_weight_current_itr_mean`：gamma；`real_consistency_effective_weight_itr_mean` 和 `virtual_consistency_effective_weight_itr_mean` 为包含外层系数的实际均值权重。
- 继续保留真实/虚拟 consistency 原始损失、生成数量、任务覆盖率、路径验证能量和 RL 有效权重，与最终回报一起判断。路径能量改善本身不等于策略性能提升。

验证命令：`python -m pytest tests -q`。测试覆盖真实 encoder/policy/critic 的 CPU 更新、奖励和动力学两条分支、支持筛选与配对输入、缓存失效、衰减边界与梯度、导出、命令行及所有环境配置的差异。它们不代替 MuJoCo 完整训练和多 seed 性能验证。
