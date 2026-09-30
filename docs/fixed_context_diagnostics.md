# 第 36 次：固定 context 复评与匹配对照

第 35 次没有支持继续扩大输入池或增加路径步数。因此本轮先区分 context 采集、表示变化与控制性能，并补齐参数匹配的无虚拟任务对照。现有第 34/35 次训练配置和评估指标保持原样；新增复评是独立协议，不能直接将其绝对回报与旧 `_expl` 指标混为一谈。

## 固定 context 的模型复评

`evaluate_fixed_context.py` 直接加载保存的 encoder、actor、decoder、Q1/Q2，不构建训练算法、不加载 dynamics、不更新任何参数。一个 checkpoint 内的 encoder 与 actor 始终配套。

默认使用测试任务，比较 200、400、499 轮模型。每任务有三种 context 来源：

1. 从离线数据无放回选取的固定 200 条 transition。
2. 第 200 轮 actor 在 z=0 下随机采集的固定 200 条 transition。
3. 第 499 轮 actor 在 z=0 下随机采集的固定 200 条 transition。

每种来源采集/抽取三个独立 context 重复。在线池只生成一次，每条采集到的 transition 使用一次；所有被评估模型复用完全相同的池。每个 context 只编码一次，得到固定 z，再使用相同的三个环境 reset seeds 进行确定性策略评估。评估过程中不追加 context、不重新采样 context，也不更新 z。

因此，“换 checkpoint、context 不变”与“checkpoint 不变、换 context 来源”可以分别比较。信息瓶颈模型使用后验均值（精度加权 Gaussian product），不是旧流程中的随机后验采样；当前 Ant-Dir/Cheetah-Vel 配置均未启用信息瓶颈。

每个任务还留出一组固定真实 probe。离线 context 各重复与 probe 的池内行 ID 严格不相交；底层文件本身若含重复 transition，行 ID 不同并不保证信息独立。根据训练 replay 的容量限制选择有效行，样本不足会报错，不隐式有放回补齐。

## 命令

同步本地代码后，在服务器项目根目录执行。下面命令先复评两个环境的 `35_global_cons_decay` seed0；不需要重新训练。CPU 运行可省略 `--gpu`。真实环境仍需要项目原有 MuJoCo 安装，工具会拒绝 placeholder 环境。

```bash
python evaluate_fixed_context.py --log-dir ./logs/ant-dir/gentle/seed0/35_global_cons_decay --epochs 200 400 499 --collector-epochs 200 499 --output-dir ./output/36_fixed_context/ant-dir/seed0/global_cons_decay --gpu 0

python evaluate_fixed_context.py --log-dir ./logs/cheetah-vel/gentle/seed0/35_global_cons_decay --epochs 200 400 499 --collector-epochs 200 499 --output-dir ./output/36_fixed_context/cheetah-vel/seed0/global_cons_decay --gpu 1
```

将路径中的 seed0 同时改为 seed1、seed2、seed3 可复评其他训练 seed。不要把 `--eval-seeds` 当作训练 seed：它们只控制评估环境的随机初始化；训练 seed 从各自 `variant.json` 读取。默认规模为每个运行 3 checkpoint × 10 测试任务 × 3 来源 × 3 context 重复 × 3 reset，共 810 条评估轨迹，外加在线 context 采集。

先做较小的端到端验证，可以使用：

```bash
python evaluate_fixed_context.py --log-dir ./logs/ant-dir/gentle/seed0/35_global_cons_decay --epochs 200 499 --collector-epochs 200 499 --context-repeats 1 --eval-seeds 0 --output-dir ./output/36_fixed_context_smoke/ant-dir/seed0 --gpu 0
```

小规模验证与正式评估使用不同输出目录，避免混淆协议。如果指定模型文件缺失，会直接报出缺失路径，绝不自动替换成其他 epoch。`--help` 无需导入 PyTorch、Gym 或 MuJoCo。

## 在不同实验间共享同一组 context

先完成上面的 global 复评，再用它作为共同采集器，比较 bank 模型：

```bash
python evaluate_fixed_context.py --log-dir ./logs/ant-dir/gentle/seed0/35_bank_path8 --epochs 200 400 499 --collector-log-dir ./logs/ant-dir/gentle/seed0/35_global_cons_decay --collector-epochs 200 499 --bank-dir ./output/36_fixed_context/ant-dir/seed0/global_cons_decay/context_bank --output-dir ./output/36_fixed_context/ant-dir/seed0/bank_path8_shared_context --gpu 0
```

这样三个来源的 context 都固定不变，可以比较不同实验本身的 encoder/actor。对 `35_bank_path0` 也可相同操作。只有 seed、任务、数据、归一化、context 大小/重复次数、采集器 checkpoint 与采集 horizon 等兼容时才允许复用；不兼容时拒绝读取缓存，需选择新的 `--bank-dir`。不要手工修改 manifest 绕过检查。

其他选项：

- `--offline-only`：仅固定离线 context，不进行在线采集。
- `--split train`：诊断训练任务；默认 `test`。
- `--data-dir /path/to/data`：移动数据后覆盖根目录，仍沿用原配置的任务、轨迹编号和 epoch。
- `--probe-size 1024`：增加固定真实 probe 数量，需要足够数据；改变后使用新的 bank 目录。
- `--save-trajectories`：额外保存实际评估 transition、Q、decoder 预测、有限轨迹折扣回报，方便逐项复核。

## 归一化、缓存和来源记录

新训练启动后将实际 `obs_normalizer.mean/var/count` 与训练/测试任务划分保存为 `observation_normalizer.npz`。只新增保存，不改变训练变换或随机数采样。复评优先使用该文件，并与训练数据重建的统计比对；明显不一致则报错。

旧运行没有该文件时，工具使用原配置选择的全部训练任务轨迹，从 observation 列一次重建 RunningMeanStd。不会混入测试任务、next observation 或只用抽样估计统计。归一化使用相同的初始计数 1e-4、方差和 1e-8 分母稳定项。读取顺序固定，与旧 glob 顺序的浮点归约可能有微小差异。

旧 checkpoint 无法单独证明训练数据从未被替换。此回退方式依赖原数据仍可用且未改变，结果中会明确标记 `reconstructed_from_current_training_data`。所有本次读取的文件会记录 SHA-256、行数和任务 ID，context bank 另有 archive 校验和；结果记录 variant 和模型 SHA-256。来源记录用于复核，不能反过来证明旧数据历史。

缓存必须同时满足协议元数据、完整的来源/任务/重复组合、配对数组形状与 archive 哈希。再次执行相同命令会复用同一 bank，不再次采集在线 context。输出只能写到训练目录之外，模型和原 `progress.csv` 不会被修改。

## 输出与解释

输出目录包含：

| 文件 | 内容 |
| --- | --- |
| `fixed_context_results.json` | 每 checkpoint/来源/任务/context 重复的回报、z、固定 probe 和真实 rollout 诊断，以及全部来源记录 |
| `returns.csv` | 便于对比的回报和固定 context 下的 z 变化；`reset_seed_std` 只是评估 reset 的波动，不是训练 seed 标准差 |
| `context_bank/manifest.json` | context 协议、采集器、数据与归一化来源、数组校验和 |
| `context_bank/contexts.npz` | 固定配对 transition、离线行 ID、独立真实 probes，无对象 pickle |
| `trajectory_diagnostics.npz` | 仅 `--save-trajectories` 时生成，包含真实 rollout 与对应预测 |

`within_task_z_rms` 是同一 checkpoint、来源、任务在不同固定 context 下的 z 离散程度；`z_l2_from_first_checkpoint` 是同一份 context 下相对于最早被评估 checkpoint 的变化。这些是表示变化的诊断，不单独证明语义漂移、坍缩或因果。

decoder 奖励误差使用未乘训练 reward_scale 的真实奖励；动力学误差比较实际 next observation，decoder 已加回 obs，不会重复加。probe 是真实任务数据，因此可以验证真实任务条件下的模型误差，**不能把它当作虚拟任务有真实奖励标签的证明**。

Q1/Q2/Q-min 的 MC 误差使用当前 checkpoint 的确定性策略真实 rollout 后缀，按训练 `reward_scale` 乘一次并做折扣。离线 probe 只记录 Q 分布和双 Q 分歧，不把离线行为轨迹的回报冒充当前策略 Q 真值。

MC 是有限 rollout tail，末端零 bootstrap，不是无限期或 target-policy smoothing 后的准确 Q 真值。结果同时记录 `terminated`、`truncated`、`horizon_reached`。旧环境有时用无附加信息的 `done=True` 表示内置时间上限，此时 `termination_cause_ambiguous=True`；`terminated` 只是保留环境报告，不能据此断言是真正吸收终止。

复评目前显式拒绝 `sparse_rewards=True`，避免悄悄改变稀疏 context/回报的语义。主要 Ant/Cheetah 配置为 dense，不受影响。评估核心兼容旧/新 Gym step 返回格式；完整命令仍使用仓库现有环境包装器。

## 参数匹配的无虚拟任务训练

五个连续环境的配置与命令见 [REAL_ONLY_MATCHED.md](../configs/interpolation-diagnostics/REAL_ONLY_MATCHED.md)。实验名为 `36_real_only_matched`。保留第 34/35 次真实 batch、网络、数据及预训练加载要求，关闭虚拟生成与虚拟 RL，同时补偿真实 consistency 的有效系数。

Ant/Cheetah/Hopper/Walker 的外层 consistency 权重为 `0.5*10/14=0.35714285714285715`；Point 的原始系数为 0.75，因此对应 `0.5357142857142857`。这些配置用于检验虚拟任务的净贡献，不承诺一定提高回报。

## 本地验证范围

执行 `python -m pytest tests -q`。新增测试覆盖真实网络 checkpoint 的严格加载、完整命令行执行与重复缓存复用、配对数据和留出 probe、归一化精度、冻结 z、奖励/动力学、手算 MC、RNG/模型模式恢复、时间上限标注，以及无虚拟任务时的真实梯度系数和训练更新。

这些 CPU/CUDA 单元与小环境集成测试不替代服务器实际 MuJoCo 复评和完整多 seed 训练。当前代码提供的是可复核的下一阶段实验工具，没有把未验证的性能假设写成新的训练默认。
