# 插值方法对照实验

这组配置用于比较 global、semantic 支持路径和非线性路径优化。每个环境的三份配置只有 `virtual_task_generation_mode` 与 `virtual_semantic_path_steps` 不同。它们使用统一后的训练设置，不复现第 28、32 或 33 次实验。

| 文件后缀 | 生成方式 | 路径优化步数 |
| --- | --- | ---: |
| `-global.json` | global，`M=2` | 0（global 不使用此参数） |
| `-semantic-path0.json` | 语义邻居图、数据支持筛选、参考直线路径 | 0 |
| `-semantic-path8.json` | 相同语义与支持规则，加非线性路径优化 | 8 |

path0 仍然沿图路径按语义弧长采样，可能跨越多条边。它不是两个任意任务之间的全局直线插值。path8 可能因验证失败回退到通过支持检查的参考路径；需结合 refinement 和 fallback 日志判断非线性优化是否实际生效。

## 固定的训练设置

| 环境 | 每次请求虚拟任务数 | 语义邻居候选数 | 虚拟 RL 初始权重 | 衰减起止轮次 | global beta |
| --- | ---: | ---: | ---: | --- | ---: |
| Ant-Dir | 4 | 2 | 0.5 | 150–300 | 2.0 |
| Cheetah-Vel | 4 | 2 | 0.5 | 100–250 | 2.0 |
| Point-Robot | 4 | 3 | 0.5 | 50–200 | 2.0 |
| Hopper-Rand-Params | 4 | 2 | 0.2 | 50–200 | 1.0 |
| Walker-Rand-Params | 4 | 2 | 0.2 | 50–200 | 1.0 |

三组均明确设置：

- `virtual_transition_use_recon_weight=false`。
- `virtual_transition_use_semantic_weight=false`，关闭 semantic 端点重构质量对 RL 样本的额外折扣。
- `virtual_transition_use_cycle_weight=false`。
- `virtual_transition_train_policy_q=true`、`virtual_transition_train_policy_bc=false`。
- `virtual_transition_use_policy_actions=false`。
- `virtual_transition_weight_schedule=linear_decay`，最终权重为 0。
- `require_pretrained_context=true`，global 也必须加载预训练 encoder/decoder，不允许缺失模型时继续使用随机初始化。

因此，虚拟 replay buffer 中写入的质量权重为 1，每个抽出的虚拟 RL 样本只乘以当轮的线性衰减权重。semantic 的端点重构筛选、共享数据支持检查仍生效；开关关闭的是连续降权，不是筛选规则。旧配置不指定 `virtual_transition_use_semantic_weight` 时默认为 true，保持已有 semantic 行为。

global 的 `beta=2` 可能产生外插样本。global 与 path0 的对比包含任务对选择、数据支持和采样分布的变化；path0 与 path8 才用于检验增加路径优化的效果。两组 semantic 训练后 encoder 及图结构可以分化，不保证全过程使用相同的边。

保持原有按任务数平均的 consistency 目标。实际生成 4 个虚拟任务、真实 meta-batch 为 10 时，虚拟部分占 `4/14`。无有效路径时跳过虚拟生成，仅保留真实 consistency。日志记录这一变化，不将“请求 4 个”当作“始终生成 4 个”。虚拟 RL 权重衰减到 0 后，虚拟任务仍可参与 consistency；没有新任务生成时，已有 replay 数据也可能继续参与尚未衰减到 0 的 RL 更新。

## 运行命令

在项目根目录运行。默认使用 seeds 0、1、2、3；仅运行一个 seed 时追加 `--seed_list 0`。实验名中的 `ablation` 可以统一替换为下一次实验编号。三组应使用相同 seed 对应的预训练文件和数据。

先运行 Ant-Dir 与 Cheetah-Vel：

```bash
python train_gentle.py ./configs/interpolation-ablation/ant-dir-global.json --exp_name ablation_global --gpu 1
python train_gentle.py ./configs/interpolation-ablation/ant-dir-semantic-path0.json --exp_name ablation_semantic_path0 --gpu 1
python train_gentle.py ./configs/interpolation-ablation/ant-dir-semantic-path8.json --exp_name ablation_semantic_path8 --gpu 1

python train_gentle.py ./configs/interpolation-ablation/cheetah-vel-global.json --exp_name ablation_global --gpu 1
python train_gentle.py ./configs/interpolation-ablation/cheetah-vel-semantic-path0.json --exp_name ablation_semantic_path0 --gpu 1
python train_gentle.py ./configs/interpolation-ablation/cheetah-vel-semantic-path8.json --exp_name ablation_semantic_path8 --gpu 1
```

其余连续任务环境：

```bash
python train_gentle.py ./configs/interpolation-ablation/point-robot-global.json --exp_name ablation_global --gpu 1
python train_gentle.py ./configs/interpolation-ablation/point-robot-semantic-path0.json --exp_name ablation_semantic_path0 --gpu 1
python train_gentle.py ./configs/interpolation-ablation/point-robot-semantic-path8.json --exp_name ablation_semantic_path8 --gpu 1

python train_gentle.py ./configs/interpolation-ablation/hopper-rand-params-global.json --exp_name ablation_global --gpu 1
python train_gentle.py ./configs/interpolation-ablation/hopper-rand-params-semantic-path0.json --exp_name ablation_semantic_path0 --gpu 1
python train_gentle.py ./configs/interpolation-ablation/hopper-rand-params-semantic-path8.json --exp_name ablation_semantic_path8 --gpu 1

python train_gentle.py ./configs/interpolation-ablation/walker-rand-params-global.json --exp_name ablation_global --gpu 1
python train_gentle.py ./configs/interpolation-ablation/walker-rand-params-semantic-path0.json --exp_name ablation_semantic_path0 --gpu 1
python train_gentle.py ./configs/interpolation-ablation/walker-rand-params-semantic-path8.json --exp_name ablation_semantic_path8 --gpu 1
```

Cheetah-Dir 只有一份 `n_vt=0` 的 control，不把三个完全不生成虚拟任务的配置作为三组插值实验。这份 control 也不能替代检验离散任务中虚拟增强作用的单独实验。

```bash
python train_gentle.py ./configs/interpolation-ablation/cheetah-dir-control.json --exp_name ablation_control --gpu 1
```

## 核对训练行为

每次运行的 `variant.json` 保存最终合并和 CLI 覆盖后的参数、实际 seed、配置来源，以及已加载 encoder/decoder 的路径与 SHA-256。比较三组时检查同一 seed 的模型哈希是否一致。

每轮训练记录以下指标。`_itr_mean` 是该轮所有梯度更新的算术平均，不是只取评估前的最后一个更新。原有图计数器描述当前缓存、或自最近一次图刷新以来的计数，不是累计整轮接受率。

| 指标 | 用途 |
| --- | --- |
| `num_virtual_tasks_itr_mean` | 实际生成的任务数量 |
| `virtual_task_generation_skipped_itr_mean` | 请求虚拟任务但没有生成的更新比例 |
| `num_virtual_transitions_added_itr_total` | 当轮累计加入 replay 的虚拟 transition 数 |
| `virtual_transition_effective_weight_sample_mean_itr_mean` | 实际参与 RL 的样本权重；没有虚拟 batch 时记 0 |
| `virtual_transition_recon_weight_current` | 此对照中应始终为 1 |
| `virtual_transition_quality_weight_sample_mean` | 有虚拟 batch 时应为 1 |
| `real_consistency_loss_itr_mean` / `virtual_consistency_loss_itr_mean` | 两类任务各自的未加权 consistency；无虚拟任务时后者记 0 |
| `virtual_consistency_fraction_itr_mean` | consistency 中实际的虚拟任务占比 |
| `virtual_semantic_task_coverage_fraction_itr_mean` | 至少连接一条有效边的真实任务比例 |
| `virtual_semantic_sampled_unique_edges_itr_mean` | 每次生成使用了多少条不同的边 |
| `virtual_semantic_supported_probe_count_mean` | 当前有效边可用的验证输入数量均值 |
| `virtual_semantic_sampled_support_unique_fraction_itr_mean` | 每个虚拟 context 中不同输入行 ID 数 / context 大小的均值，不保证这些行对应不同原始 transition |
| `virtual_semantic_validation_energy_ratio_mean` | 最终边的验证能量 / 参考直线能量；path0 应约为 1，回退路径也为 1 |

`virtual_semantic_sampled_quality_mean` 仍记录原始端点质量作为诊断，即使它没有用于 RL 加权。不要将它误读为实际样本权重。图覆盖率也不保证在真实任务流形上。

## 导出可分析的结果

绘图时加上 `--export-run-data`，会在原有图和均值 CSV 旁额外保存 `*_seeds.csv` 与 `*_runs.json`：前者是未平滑、未插值的各 seed 原始标量记录，后者包含实际匹配到的运行配置。缺失的运行配置会明确记为 null，不用当前源码配置代替。

```bash
python plot_tb_seed_average.py --root ./logs/ant-dir/gentle --experiment ablation_global --experiment ablation_semantic_path0 --experiment ablation_semantic_path8 --tag Return/AverageReturn_all_test_tasks_expl --smooth-window 10 --export-run-data --output ./output/ablation/ant-dir-return.png
```

诊断指标也可以用同一脚本导出，替换 `--tag` 和输出文件名即可。通过 `--list-tags` 查看实际名称；例如任务生成数的轮均值通常是 `Other/num_virtual_tasks_itr_mean`。

```bash
python -m pytest tests -q
```

测试覆盖最终 CLI 配置的一致性、线性权重与 replay 权重、path0 不创建优化器、路径约束、真实 actor/critic/encoder 更新、日志轮均值以及原始 seed 数据导出。它们不替代完整 MuJoCo 离线数据训练和最终回报评估。
