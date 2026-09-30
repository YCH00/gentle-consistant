# 第 36 次：与第 34/35 次参数匹配的无虚拟任务对照

实验名统一为 `36_real_only_matched`。它用于测量整套虚拟任务训练的净贡献，并不是新的语义插值算法。已有 `gentle_base` 的 batch、预训练模型记录存在差异，不能替代这个对照。

五个配置均从第 34 次对应的 `interpolation-ablation/<env>-global.json` 派生。网络、真实 RL batch、context batch、离线数据、训练轮数、奖励缩放以及预训练模型强制加载要求全部保留。原有第 34/35 次配置不改动。

改动仅为：`n_vt=0`；虚拟 transition 初始和最终权重均设为 0；虚拟 consistency schedule 显式设为 `constant`；真实 consistency 外层权重做归一化补偿。

原配置的 consistency 为 `c * (sum(real_losses) + sum(virtual_losses)) / (R + V)`。关闭虚拟任务后使用 `c_matched * mean(real_losses)`，其中 `c_matched = c * R / (R + V)`。因此每个真实任务项的系数仍然是 `c / (R + V)`，而不是在移除虚拟项后被动增大。

这里 `R` 是 `meta_batch`：训练循环选择多少个任务条目，就有多少个真实 consistency 项。即使有放回采样产生重复任务 ID，也不按唯一任务数重算。当前五个配置均为 `R=10`、原 `V=4`。

| 环境 | 原外层权重 | 本次外层权重 |
|---|---:|---:|
| Ant-Dir | 0.5 | 0.35714285714285715 |
| Cheetah-Vel | 0.5 | 0.35714285714285715 |
| Point-Robot | 0.75 | 0.5357142857142857 |
| Hopper-Rand-Params | 0.5 | 0.35714285714285715 |
| Walker-Rand-Params | 0.5 | 0.35714285714285715 |

优先补跑 Ant-Dir 和 Cheetah-Vel，在服务器同步本地修改后，从项目根目录执行。每条默认运行 seed 0、1、2、3；GPU 编号可按空闲设备修改。

```bash
python train_gentle.py ./configs/interpolation-diagnostics/ant-dir-real-only-matched.json --exp_name 36_real_only_matched --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/cheetah-vel-real-only-matched.json --exp_name 36_real_only_matched --gpu 1
```

其余环境配置同时提供，需研究对应环境时执行：

```bash
python train_gentle.py ./configs/interpolation-diagnostics/point-robot-real-only-matched.json --exp_name 36_real_only_matched --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/hopper-rand-params-real-only-matched.json --exp_name 36_real_only_matched --gpu 0
python train_gentle.py ./configs/interpolation-diagnostics/walker-rand-params-real-only-matched.json --exp_name 36_real_only_matched --gpu 0
```

Cheetah-Dir 的既有对照已经关闭虚拟任务，因此不新增同名虚拟任务消融。

运行后需核对真实 consistency 有效权重与上表一致，虚拟任务数、虚拟 transition 加入量、虚拟 critic/actor batch 数均为 0。预训练模型路径和 SHA256 应与对应 seed 的第 34/35 次保存记录匹配；配置保留了加载约束，但无法阻止服务器上的 checkpoint 文件被替换。

关闭虚拟采样会改变随机数消耗，因此相同 seed 不保证与有虚拟任务实验具有相同的训练轨迹前缀。这里匹配的是超参数和真实 consistency 系数，并不要求两次训练逐步使用完全相同的随机样本。

与 global 相比，这个对照同时移除虚拟 RL 和虚拟 consistency，用于估计整套虚拟训练的净贡献；若后续要区分二者，仍需要各自单独关闭的对照。与 semantic 相比，只有每步都成功生成原定 4 个虚拟任务时真实 consistency 的原系数才严格相等；第 34/35 次 Ant/Cheetah 日志满足这一条件。改变 `meta_batch`、原 `n_vt` 或发生生成失败时应重新核对这一匹配关系。
