# 离线元强化学习基线复现：统一实验协议与代码修改交接

版本：`shared-main-v1`；整理日期：2026-10-10；复核日期：2026-10-11。

本文用于交接 FOCAL、CORRO、CSRO、GENTLE、UNICORN-SS 和 ER-TRL 的代码适配与复现工作。目标是在统一实验条件下比较各方法，保留各方法自身的算法设计。**复现原算法，并按照本文协议重新评估，不等于复刻原论文的全部实验设置或数值。**

参考仓库：`D:/Desktop/offline meta RL/gentle-consistant`。

本次核对的参考提交：`075df392bbaad8177339c41b700d567ef550e94f`。路径可因机器变化，协议版本和参考提交必须记录。本协议针对 `configs/` 下六个环境的根配置及其默认值，不以历史消融目录中的配置代替主协议。

本文明确区分三类信息：

- **已核对的实现事实**：参考仓库中的配置和实际数据流，例如任务划分、批大小和累计 context 评估。
- **本协议新增的统一约定**：例如使用最终 checkpoint、跨训练 seed 的样本标准差、运行清单和缺失数据检查。这些是后续所有方法共同遵循的报告规则，不表示旧代码已经实现。
- **运行时必须补全的事实**：实际数据文件、任务参数清单、基线源码版本、依赖版本及运行结果。不能把计划配置写成已验证结果。

## 0. 直接交给新对话的提示词

复制以下内容，并填写方法名、目标代码路径和可用的数据路径。建议每个对话仅负责一个基线。

```text
请阅读 BASELINE_REPRODUCTION_HANDOFF.md，并据此实际修改目标仓库，完成统一实验协议适配。

本次方法：<FOCAL / CORRO / CSRO / GENTLE / UNICORN-SS / ER-TRL>
目标仓库：<绝对路径；若就是当前工作区，可写“当前工作区”>
参考仓库：D:/Desktop/offline meta RL/gentle-consistant
共享离线数据：<路径；暂未挂载可明确写“暂未挂载”>
共享实验清单/normalizer/context bank：<已有则给出路径，没有则先生成可导出的清单工具>
运行资源：<本机或服务器、CPU/GPU、已有 Python/MuJoCo 环境>

请先阅读目标仓库的 AGENTS.md 等约束并检查已有修改，核对论文/作者实现和训练评估入口，
然后实际修改配置、数据适配和评估代码，保留该方法的核心算法。
以交接文档为共同协议，不把本文方法的 consistency、虚拟任务或 TD3+BC 强加给其他基线。
重点检查 task manifest、真实数据文件、归一化、终止标记、训练更新次数和两种主评估协议。

交付修改后的代码、六环境配置、配置映射表、适当的验证、可执行命令及 REPRODUCTION_REPORT.md。
命令需对应本仓库真实 CLI，包含 seed 0/1/2/3、评估和结果汇总方式。
若缺少数据或 MuJoCo，先完成不依赖它们的修改和验证，并准确记录尚未完成的真实环境验证。
不要仅给建议或方案；不要把 smoke test、随机策略或模拟数据结果当作正式实验结果。
本次默认先完成代码适配和小规模验证；完整训练若已在本对话授权，则继续执行并记录结果。
```

除非用户在新对话明确改变任务，不修改参考方法的科学设计，也不替其他基线独立改变本协议。确有不兼容时，先完成可以独立进行的工作，在报告中写明具体冲突及影响；常规路径适配和兼容性修复直接处理。

## 1. 统一什么，保留什么

| 项目 | 统一要求 | 保留的算法差异 |
| --- | --- | --- |
| 环境与任务 | 相同环境实现、任务参数、任务 ID、划分和 episode horizon | 不改成基线原仓库中同名但不同定义的环境 |
| 离线数据 | 使用同一批原始文件及训练/测试隔离规则 | 原方法的训练采样、负样本构造、合法的数据增强可保留并记录 |
| 预处理 | 相同 observation normalizer、动作坐标和奖励单位 | 原方法必要的辅助目标变换需要显式记录 |
| 主要容量与优化 | 对应 MLP 的层数/宽度、latent 维度、真实 RL batch、主要学习率按本文设置 | 原生后验、聚合器、辅助网络、特殊优化器及损失结构保留 |
| 训练预算 | 500 个外层迭代，每轮 400 个主训练更新 | 原生辅助更新和必要预训练单独计数，不宣称总算力相同 |
| 测试预算 | 相同测试任务、context 信息预算和回报统计流程 | 自主采集协议保留各方法原生采集机制，尤其 CSRO |
| 模型选择 | 主表统一最终 checkpoint 499；相同训练 seeds | 不按测试最高回报挑 checkpoint 或调参 |

**禁止直接将参考仓库的完整配置复制给所有基线。** 其中大量字段属于本文方法，例如 consistency、虚拟任务生成、虚拟转移加权、冻结 decoder，不是公共实验条件。

公共设置与方法定义存在实质冲突时，保留定义该方法的机制，并在配置映射表中标为例外。例如，原方法使用概率后验，不因参考仓库 `use_information_bottleneck=False` 就删除其后验或 KL 项；需要专用网络的算法也不必改写为同一个 mean-pooling encoder。任何容量或预算例外均需明确量化，不应悄悄使用另一套设置。

### 六个基线的适配边界

| 方法 | 需要保留和核对的内容 | 常见错误 |
| --- | --- | --- |
| FOCAL | 原生任务度量/距离目标、上下文推断和离线策略学习 | 改成生成式重构 encoder；只保留名字 |
| CORRO | 转移级对比目标、条件负样本构造及上下文推断 | 将其替换成普通任务标签分类；删去其目标依赖的 transition 字段 |
| CSRO | 原生互信息约束、任务推断、专用 context 采集机制 | 为“公平”强制使用本文的零 latent 采集器，移除原方法重要组件 |
| GENTLE | 原始生成式任务表征、重标注与下游训练流程 | 将当前修改版关闭几个开关，直接命名为原版 GENTLE |
| UNICORN-SS | 论文中的 self-supervised 变体及其目标组合 | 跑 SUP/oracle 变体，或将真实任务参数输入模型 |
| ER-TRL | 原生熵正则化及其辅助模型/优化过程 | 仅移植 encoder 外壳，遗漏定义该方法的正则化 |

每个新对话应依据论文和作者代码核对具体实现，并记录 URL、commit 和必要修复。文末给出论文入口；本地尚无这些基线的完整实现，不能预填未经核对的损失系数或代码版本。

原版 GENTLE 与“本文骨架的无新增组件版本”应分别命名。后者可以用于组件消融，建议标为 `matched-backbone`，只有逐项核对与原版一致后，才可当作原版 GENTLE。重点核对预训练、decoder 是否冻结、重标注配比、encoder 更新和策略目标。

## 2. 环境、任务与数据配置

### 2.1 六个环境的统一设置

表中 H 为单条轨迹的最大步数，轨迹数为每个任务应读取的离线文件数量，checkpoint 指生成数据的 SAC 模型训练步数。

| 配置名 | 任务变化 | 训练/测试任务数 | H | 每任务轨迹数 | 数据 SAC checkpoint |
| --- | --- | ---: | ---: | ---: | ---: |
| `point-robot` | 目标位置，各坐标独立采样于 `[-1, 1]` | 10 / 10 | 20 | 100 | 100000 |
| `ant-dir` | 目标运动方向，均匀采样于 `[0, 2π]` | 10 / 10 | 200 | 50 | 1000000 |
| `cheetah-vel` | 目标速度，均匀采样于 `[0, 6]` | 10 / 10 | 200 | 50 | 1000000 |
| `cheetah-dir` | 两个方向 `{-1, +1}` | 2 / 2，同一组任务 | 200 | 100 | 1000000 |
| `hopper-rand-params` | 质量、惯量、摩擦和阻尼 | 10 / 10 | 200 | 50 | 1000000 |
| `walker-rand-params` | 质量、惯量、摩擦和阻尼 | 10 / 10 | 200 | 50 | 1000000 |

连续任务使用任务采样种子 **1337**，先构造 20 个任务，ID `0–9` 用于训练，`10–19` 用于测试。任务集合在所有训练种子和算法间固定。Cheetah-Dir 只包含 ID `0,1`，训练测试均使用它们。

Hopper/Walker 的随机化规则为：质量、惯量、摩擦参数的每个元素乘以 `1.5**u`，阻尼的每个元素乘以 `1.3**u`，其中各扰动按参考实现从 `U[-3,3]` 采样。最终任务清单必须保存实际参数数组，不能只保存这些范围。

必须注意：

1. 参考 Cheetah-Vel 的实际代码范围为 `[0,6]`，不要沿用旧 docstring 中的 `[0,2]`，也不要使用 placeholder 的不同范围。
2. Point-Robot 是上述方形区域任务，不是其他实现中常见的稀疏半圆目标任务。
3. Walker 配置中的 `randomize_tasks=0` 不表示物理参数固定；以实际 wrapper 的采样结果为准。
4. 相同随机种子不足以保证相同任务。旧 NumPy 随机数接口与 `default_rng` 的序列不同，环境构造顺序也会影响结果。优先导出并加载同一份 `task_manifest`，核对实际任务参数。
5. 环境名称和参数范围相同也不够：奖励公式、观测、reset 分布、动作缩放、终止逻辑和 MuJoCo/XML 必须一致。
6. 当前连续任务是同分布抽样后的不重叠训练/测试集合，**不是预设的范围外任务划分**。Cheetah-Dir 不用于证明未见任务泛化。外推实验需另建所有方法共享的任务清单和协议。

### 2.2 环境后端检查

参考 `rlkit/envs/__init__.py` 会根据 `MUJOCO_PY_MJPRO_PATH` 选择不同导入分支：`mjpro131` 分支用于随机物理参数环境；其他 MuJoCo 路径用于相应控制环境；缺少该变量可能导入 `non_mujoco_*` 占位实现。

正式评估必须确认加载的是真实环境，记录类名、源文件、MuJoCo/Gym 版本和 XML 来源。**占位环境不能用于正式回报或作为真实环境验收通过的证据。** 若目标仓库改用较新 Gym API，可以适配接口，但不能直接把 Gymnasium 新版本同名任务视为等价环境。

参考依赖文件中的 `gym==0.17.2`、`numpy==1.23.5`、`torch==2.4.1` 可作为兼容性线索；它们不构成已经验证过的完整安装镜像。Hopper/Walker 与其他 MuJoCo 环境可能需要不同运行环境，分别记录即可。

### 2.3 离线文件选择与数据来源

所有算法共用既有数据，**不要为每个基线重新收集一套数据**。将参考配置中的服务器绝对路径改为可覆盖的 `data_dir`，保持文件选择语义不变。

每环境的数据目录结构：

```text
<data_dir>/
  goal_idx0/
    trj_evalsample0_step1000000.npy
    trj_evalsample1_step1000000.npy
    ...
  goal_idx1/
    ...
```

Point-Robot 的文件 checkpoint 使用 `100000`。轨迹编号为 `0` 到 `n_trj-1`。训练/测试分别读取对应任务目录，`train_epoch` 与 `eval_epoch` 均使用表中 checkpoint。

原始文件每行保存 `[observation, action, reward, next_observation]`；可能采用 NumPy object array，加载后需核对字段形状和数值。不要将整个二维数组按未知维度直接切片；先确认实际文件布局。不得混入更早 checkpoint、额外轨迹、测试任务或其他质量的数据。

参考数据生成流程为每任务训练 SAC，再以给定 checkpoint 采集轨迹；采集代码使用随机策略动作 `sample=True`，额外动作噪声默认 0。数据生成环境 seed 为 1。晚期 checkpoint 或目录名含 `expert` 不足以证明数据已经达到专家质量，论文中的质量描述需要实际数据统计支持。

加载时应生成 `dataset_manifest`，至少包括：环境、任务 ID/划分、相对文件路径、SHA-256、实际行数、字段维度、数据 checkpoint、每任务总行数。按固定顺序读取文件。

参考旧 loader 可能静默跳过不存在的文件；新适配必须检查文件完整性并明确报错，不能少加载若干文件后仍声称数据相同。表中的轨迹数量是期望值，实际 transition 数由文件决定，不能用 `n_trj × H` 或 replay 容量代替实测数量。

### 2.4 终止标记与预处理

**终止标记。** 原始文件不包含明确的 `terminated/truncated` 标签。参考 loader 将每个轨迹最后一行设为 terminal=1，其余为 0。本协议沿用此行为，记录为 `terminal_mode=legacy_episode_end`，包括可能由时间上限结束的轨迹。

这是一项兼容性约定，不代表能从旧文件恢复真实终止原因。不能只为某个基线修正 timeout bootstrap；若以后调整该规则，应统一升级协议并重跑涉及的方法。不得凭空生成已知的真实终止/截断标签。

**状态归一化。** 使用全部训练任务、已选文件中的 current observations 计算同一个 normalizer，不混入 next observations 或测试数据。参考 `RunningMeanStd` 初始 `mean=0, var=1, count=1e-4`，以总体方差更新，变换为：

```text
normalized_obs = (obs - mean) / sqrt(var + 1e-8)
```

同一组统计用于 current/next observations、离线 context 和在线评估观测。保存并共享 `mean/var/count` 及文件校验和，避免不同方法独立估计出不同统计。参考新运行文件为 `observation_normalizer.npz`。禁止重复归一化；共享 context bank 的坐标系以其 manifest 为准。

**动作。** 策略接口采用归一化动作坐标 `[-1,1]`，环境 wrapper 映射到实际动作边界。核对数据已有的动作单位，避免二次缩放；Point-Robot 的物理动作边界为 `[-0.1,0.1]`。

**奖励。** context、重构标签及评估回报使用原始奖励。下游 Bellman 更新的公共 `reward_scale=5`，只乘一次；原方法若有额外奖励变换或与熵系数耦合的设计，应记录其实际计算式，不可隐式二次缩放。

**数据隔离。** 测试任务数据仅用于规定的适应/评估；不用于 encoder、策略、辅助模型、normalizer 的训练或超参数选择。任务 ID 可以组织 replay 和训练任务正负样本，不能作为测试时真实任务条件输入模型。

**Context 字段。** 共同原始数据始终保留完整 `(s,a,r,s')`。参考方法在奖励变化环境中使用 `(s,a,r)`，在 Hopper/Walker 中使用 `(s,a,r,s')`；这是参考 encoder 的输入选择，不是从共享数据中删除 `s'` 的要求。各基线按原生目标选择必要字段并记录，尤其不能删掉 CORRO 等方法所依赖的信息，也不要未经核对增加原方法此前不使用的 oracle 信息。

## 3. 网络、优化与训练预算

### 3.1 公共数值

| 环境 | 对应 MLP 隐藏层 | latent 维度 | meta-batch | 每任务条目的 RL batch | 每任务条目的训练 context | γ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Point-Robot | `64,64,64` | 5 | 10 | 512 | 512 | 0.9 |
| Ant-Dir | `256,256,256` | 5 | 10 | 512 | 512 | 0.99 |
| Cheetah-Vel | `256,256,256` | 5 | 10 | 512 | 512 | 0.99 |
| Cheetah-Dir | `256,256,256` | 5 | 4 | 512 | 512 | 0.99 |
| Hopper-Rand-Params | `256,256,256` | 10 | 10 | 512 | 256 | 0.99 |
| Walker-Rand-Params | `256,256,256` | 10 | 10 | 512 | 256 | 0.99 |

公共优化与预算：

```yaml
# 协议语义示例；目标仓库需映射为自己的真实配置键。
protocol_id: shared-main-v1
task_sampling_seed: 1337
training_seeds: [0, 1, 2, 3]
outer_iterations: 500
main_updates_per_iteration: 400
total_main_updates: 200000
policy_lr: 0.0003
critic_lr: 0.0003
encoder_lr: 0.0003
common_optimizer: Adam
real_rl_batch_per_task_entry: 512
reward_scale_for_rl: 5.0
main_checkpoint_iteration: 499
saved_iteration_indices: [0, 200, 400, 499]
cross_training_seed_std_ddof: 1
```

解释与边界：

- `meta_batch=10, batch_size=512` 表示每次主更新中真实 RL 样本共 5120 条；Cheetah-Dir 为 2048 条。不是所有任务合计只取 512 条。
- 任务数量足够时，无放回抽取 meta-batch；Cheetah-Dir 从 2 个任务有放回抽取 4 个条目。重复条目独立抽样，不应被去重成 2 个条目。
- 对比/度量目标仍按真实训练任务 ID 构造正负关系，不能把同一任务的重复条目误当成负任务。核对原生损失归一化，不能因调整 batch 形状而无意改变各目标的相对系数。
- 参考 context mini-batch 与 context batch 相同。不要把训练 context 长度误用成测试 context 长度，测试规则见第 4 节。
- 对应网络采用上述公共层宽和主要学习率；必要的原生结构、激活、概率建模和辅助学习率保留并记录。方法特有的损失系数、温度、负样本数等优先采用作者实现的相应默认值，写明来源。
- 200000 是主训练循环调用次数，不保证每个方法的 actor、critic、encoder、辅助网络各更新 200000 次。保留原生必要的更新顺序/比例，逐类记录实际 optimizer step 数和预训练成本。
- 显存不足可在保持等价损失缩放、有效 batch 和更新次数的前提下使用梯度累积。不能静默减小 batch、latent 维度或训练时长。
- 训练种子为 0/1/2/3。参考 `default.py` 的 `seed_list=[0]` 会被训练 CLI 默认值覆盖，不能只看默认文件。
- 环境任务集合固定，训练种子控制初始化、抽样及相应环境随机性。分别保存 Python/NumPy/Torch/CUDA 等随机状态；断点恢复需恢复优化器、调度器和主更新计数。
- 元训练参数更新只使用固定离线数据及算法自身允许的合成数据。训练期间的测试 rollout 可以用于记录指标，但不能回流训练、更新统计或据此挑超参数。

### 3.2 不应统一移植的参考字段

下列字段或机制属于参考算法，不是所有基线的公共要求：

- TD3+BC 的行为克隆项、target smoothing、双 Q 具体更新方式及对应超参数。
- GENTLE/本文的 dynamics ensemble、任务自编码器预训练和 policy relabel。
- 本文的冻结 decoder、真实任务循环一致性、虚拟任务恢复和虚拟 RL 权重调度。
- `use_information_bottleneck=False`、特定 encoder 的 tanh/mean pooling。

参考配置中 `policy_freq=2` 虽然存在，但当前主更新实现未按它延迟 actor 更新。不能根据一个未生效的键，宣称参考方法每两步才更新 actor，更不能把该值机械移植为所有基线的约束。

参考 `collect_data` 在主训练中从既有离线 buffer 抽取数据，函数名并不意味着允许在线采集训练数据。目标实现可以保留自身合理的 replay 管理方式，但共同的原始可用数据、batch 和训练预算必须一致，并记录实际采样来源。

当前参考根配置的虚拟任务生成模式为 `semantic`，不等同于论文早期讨论的简单线性插值草案；五个连续任务配置启用虚拟任务，Cheetah-Dir 不启用。复现其他基线不需要复制这些设置，也不要借本次适配更改本文方法。

### 3.3 模型选择与调参

主表固定使用 500 轮训练结束的 checkpoint，即从零计数的 **499**。这是本协议确定的报告规则，旧日志里存在中间评估值不意味着允许选测试最优 checkpoint。

没有预设的独立验证任务划分时，公共参数固定，方法特有参数使用已记录的作者默认设置。若后续安排调参，必须先定义仅依赖训练侧数据的共同选择流程和预算，再对所有参与方法一致执行；不能隐式用测试任务做验证集。

中间 checkpoint 0/200/400 可用于曲线和诊断，不替代主表模型。若原仓库 checkpoint 按 gradient step 命名，应明确映射：外层迭代 i 保存于完成 `(i+1) × 400` 次主更新后，最终为 200000 次。

## 4. 两种主评估协议

主评估分别报告 `shared-main-v1/given_context` 和 `shared-main-v1/self_collected`，不要混为一个指标。这两个名称是交接文档的统一标识，需要目标实现保存到输出。

### 4.1 Context 的初始来源

**Given-context。** 从目标测试任务的固定离线数据 buffer 提供 context。策略不需要先在线探索；这些测试数据仍不能用于参数更新。

**Self-collected。** 在目标任务中先采集一个固定初始 context pool，Point-Robot 恰好 20 条新 transition，其他环境恰好 200 条。提前终止后 reset 继续采集，达到预算立即停止，允许采集阶段的最后一条轨迹不完整。

参考方法用当前策略在 `z=0` 下随机采集。其他基线保留原方法的采集机制，尤其 CSRO 的专用采集器；可按其原生机制更新采集阶段的内部 belief/hidden state，但不进行测试任务上的梯度训练，也不超过相同 transition 预算。不得输入 oracle 任务参数或调用该测试任务的专家策略。

如果原方法没有专用采集器，使用该方法的先验任务条件和原生策略采集，并记录先验形式、随机/确定性动作模式。不要假设所有方法的“先验”都是同一个零向量。

因此，自主采集主结果衡量的是统一预算下包含采集过程的完整方法。专门比较 encoder 的行为鲁棒性时，另外采用第 5 节的共享固定 context。

### 4.2 采集后的共同评估流程

对每个训练 seed、checkpoint、测试任务和协议，执行：

```text
clear task inference state
pool = offline task buffer                         # given_context
   或 collect_once_with_native_collector(B)         # self_collected
clear evaluation inference state, retaining pool
C = empty
evaluated_steps = 0
episode_returns = []

while evaluated_steps < 600:
    C_new = sample H transitions from pool WITH replacement
    C = concatenate(C, C_new)
    reset inference accumulator and infer from the entire C
    fix this task condition within the next episode
    run a complete deterministic-action episode until done or H
    record its raw, undiscounted return and actual length
    evaluated_steps += actual episode length
    do not append this evaluated episode to C or pool

task_score = mean(episode_returns)
```

其中 H 为环境 horizon，B 为初始采集预算，Point-Robot 为 20，其他环境为 200。每个任务 `num_evals=1`。

以下细节必须保持一致：

1. 每条评估轨迹之前，从同一个 pool **有放回**抽 H 条，再追加到累计 context；不是每次丢弃旧 context，也不是持续收集新 context。
2. 采集结束后清空用于评估的 posterior、hidden state 和推断缓存，保留已采集的 pool。每条轨迹前，基于全部累计 context 从初始推断状态重新计算任务条件；不能把整个 C 再次累积进已包含旧 C 的后验。等价的增量计算只有在确认不重复计入证据时才可使用。单条轨迹期间 task condition 固定。
3. actor 采用确定性评估动作，例如 Gaussian policy 的 tanh-mean；原本确定性的 actor 直接输出动作。确定性动作不等于必须删除概率 encoder，任务后验的推断规则需保留原生定义并记录。
4. 600 是评估步数的停止阈值，不强行截断最后一条完整轨迹。因此实际步数可能略大于 600。
5. 自主采集的初始 pool 只生成一次；其采集回报不计入主评估回报，采集交互与后续评估交互分别记账。
6. 没有提前终止时，Point-Robot 评估 30 条轨迹，context 长度依次为 20、40、…、600；其他环境评估 3 条轨迹，依次为 200、400、600。提前终止时轨迹数更多，累计 context 甚至可超过 600。
7. 重复抽样导致编码序列变长，不表示获得更多新的任务反馈。应记录 source pool 大小、抽样行 ID、累计 context 长度、实际轨迹数和交互步数。
8. 若某原生模型必须使用不同的时序 context 组织，应明确报告与此统一协议的兼容方式；不能静默换评估流程。可另报原生评估协议，但其数值不能混入统一协议主表。

参考代码在最后一次评估还会为可视化额外采集 `5 × online_sample_num` 条 transition，生成 400 个 latent 样本。这是诊断开销，**不是主评估适应预算**，不要让它流入上述主结果。

### 4.3 回报、随机种子与统计

- 单轨迹回报为原始奖励的非折扣和，不乘训练 `reward_scale=5`。
- 先对同一任务的完整轨迹回报取平均，再对同一训练 seed 的测试任务取平均，最后在 4 个独立训练 seed 上计算均值和标准差。
- 跨训练 seed 的标准差统一用样本标准差 `ddof=1`，并在汇总脚本和附录注明。这是统一报告约定。
- reset seed、context 重复和同一模型的多条轨迹不是独立训练运行，不能替代 4 个训练 seed。
- 缺失运行、崩溃、NaN 均须显式列出，不静默丢弃或替换为新 seed。少于 4 个成功运行时标明完成数，不冒充完整结果。
- 不直接平均不同环境的原始回报。若新增跨环境归一化汇总，需要预先定义共同参照和公式。
- 相同训练 seeds 可报告方法间配对差值；置信区间方法需明确。小样本结果不能仅凭均值排序宣称显著性。

### 4.4 主评估的随机性记录

以下是本协议新增的复现约定，本文方法和各基线均需采用或明确报告差异；不是对历史日志中随机状态的推断。独立复评不能仅凭训练 seed 就承诺复现旧训练过程中的末轮数值。

为各方法共享生成 `evaluation_seed_manifest.json`，至少独立规定 collector、context 抽样、posterior 推断和 rollout 环境 reset 的随机种子。统一种子计算方式为：

```python
import hashlib
import json

def evaluation_seed(protocol_id, env_name, training_seed, task_id, stage, index):
    # stage: collector / context / posterior / rollout
    # collector 使用 index=0；其余 index 为从 0 开始的评估轨迹编号。
    key = [protocol_id, env_name, int(training_seed), int(task_id), stage, int(index)]
    payload = json.dumps(key, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
```

种子键不含方法名和 checkpoint，以便跨方法/阶段比较使用同一随机计划。采集前设置 collector 的随机状态；每条评估轨迹的抽样、后验推断和环境 reset 分别使用对应阶段种子。Context 行索引统一用局部 `numpy.random.RandomState(seed).choice(pool_size, size=H, replace=True)` 生成，pool 的行顺序需稳定。涉及 Python、Torch/CUDA 或环境自身随机生成器时，要显式设置并记录，不能只调用 `np.random.seed`。

共享随机种子不意味着不同策略会产生相同轨迹，也不意味着不同网络会使用相同噪声序列；固定输入诊断仍以真实共享的 context 数组为准。保存实际使用的 seed/行索引及 RNG 规则版本。训练内评估应隔离或保存恢复训练 RNG，避免改变评估次数就改变后续训练抽样；断点独立评估使用同一清单。

## 5. 共享固定 context 诊断：与主评估分开

该协议服务论文中 behavior/task shift 的分析。参考实现位于 `evaluate_fixed_context.py`、`evaluate_fixed_context_suite.py`、`rlkit/torch/fixed_context_eval.py`，说明见 `docs/fixed_context_diagnostics.md`。

适配基线时需要使其能接收外部 context bank 并加载自己的 encoder/actor；不能直接把其他算法的 state_dict 塞给本文网络。当前参考工具不是已经支持全部六个算法的通用 loader。

统一诊断设置沿用参考工具默认值：

| 项目 | 设置 |
| --- | --- |
| 协议 | `fixed-context-v1`，独立于两种主评估 |
| 模型 checkpoint | 200、400、499 |
| 每任务 context 长度 | 200，与主评估累计 context 区分 |
| 来源 | 离线 buffer、共同采集器 checkpoint 200、共同采集器 checkpoint 499 |
| 每来源 context 重复 | 3 |
| 每个固定 context 的 rollout reset seeds | 0、1、2 |
| 默认 bank seed | 3600；保存最终实际使用值及任何派生规则 |
| 真实机制 probe | 每任务默认 256 条，离线行 ID 与 context 分离 |

诊断要求：

- **同一训练 seed 的所有方法使用完全相同的 context 数组**，由预先指定的同一采集器生成并缓存。记录采集器方法、checkpoint/hash、动作模式和数据来源。不能让每个方法各自采集后仍称“共享 context”。
- 当前参考共同采集器采用零 latent 的随机 actor；正式跨算法比较前固定其身份。不可根据某个被评估方法的测试结果挑选采集器。
- 离线 context 按参考工具无放回抽样，probe 与各离线 context 的行 ID 分离；不足时明确失败，不自动改为有放回。行 ID 不同不保证原始文件里不存在重复 transition。
- 每份 context 只编码一次，所得任务条件在对应各次 rollout 中固定；不追加 context、不重新采集或更新网络参数。
- 参考工具对信息瓶颈模型采用后验均值。目标基线若定义了兼容的后验均值，统一使用并注明；需要不同原生推断时声明具体规则和随机性，不能随意把分布模型改成参考 deterministic encoder。
- 复用 bank 前核对任务清单、数据、坐标系、normalizer、来源、seed 和数组哈希。不能手改 manifest 绕过兼容性检查。
- 不同方法的 latent 坐标系通常不对齐，不能直接比较跨模型向量距离。t-SNE 仅作辅助可视化；结合各模型内部同任务跨 context 稳定性、任务区分、可用的真实机制预测及回报指标解释。
- 没有 decoder 的基线无需为了指标强行增加 decoder；涉及 probe 的指标注明适用方法。对虚拟任务的生成反馈不能声称拥有真实环境标签。

该诊断和额外的范围外任务实验不是完成主表的前置训练条件。先把两种主评估跑通，再在共享 bank 就绪时复评；两种协议的绝对回报不能直接混用。

## 6. 修改代码的执行顺序与验收

### 6.1 先审计，再实际修改

1. 阅读目标仓库说明/约束，检查工作区已有变更；记录方法、论文版本、源码 URL/commit，不覆盖用户修改。
2. 定位配置合并、环境构造、数据 loader、normalizer、训练更新、context 推断、collector、评估和 checkpoint 入口。
3. 建立配置映射表：`协议字段 → 原字段/代码位置 → 修改后生效值 → 验证证据/例外原因`。以运行时解析值和调用路径为准，不只修改一个未被读取的 JSON。
4. 增加独立的共享协议配置/入口，尽可能保留原生运行模式。六环境覆盖值显式化，禁止不同对话各自推断不同默认值。
5. 接入共同数据、任务清单、normalizer 和第 4 节评估；保留原算法训练目标及原生 collector。
6. 在训练启动时打印并保存最终配置、任务和数据指纹、有效 batch、预期更新次数，及时拒绝错误数据或占位环境。
7. 执行针对修改的验证并交付真实 CLI 命令。缺数据或 MuJoCo 时完成可以进行的工作，准确记录未覆盖范围。

无需把所有仓库重构成相同代码结构。接口适配优先，避免将复现工程变成新的算法开发。

### 6.2 至少验证这些关键行为

对涉及数据和评估语义的改动，使用小样例/受控环境验证，而不是仅检查程序能启动：

- **配置生效**：六个环境解析后的表中参数一致；CLI 路径覆盖不改变其他协议字段；seed 列表正确。
- **数据正确**：文件缺失会被发现；任务划分无泄漏；normalizer 仅来自训练 current observations；动作/奖励没有二次变换；terminal 与公共规则一致。
- **更新预算**：短运行计数正确，meta-batch 和 per-task batch 没有混淆；必要原生目标得到梯度；测试/诊断不会更新参数。
- **主评估语义**：用可控 episode 长度验证累计 context、有放回抽样、预算内初始采集、整轨迹停止和评估轨迹不回流 context。另覆盖提前终止情况。
- **推断与保存**：checkpoint 能重新加载，encoder 与 actor 配套，最终轮编号映射正确；必要时验证概率后验和 collector 状态重置。
- **结果汇总**：手算小样例核对“轨迹→任务→训练 seed”的平均顺序、样本标准差和缺失运行处理。
- **真实 smoke**：若具备真实数据和环境，至少一个环境执行小规模加载、若干真实梯度更新、保存/加载及两种评估。标注实际更新数，不把 smoke 的短预算写进正式配置。

小样例测试只验证实现逻辑，不证明 MuJoCo 兼容性或正式性能。只做过 CLI/静态检查时，不写“复现已完成”。

### 6.3 各基线交付清单

建议目录结构如下；可顺应目标仓库调整命名，但交付内容不能缺失：

```text
configs/shared_protocol/             # 六环境配置或等价单一配置+覆盖
scripts/                            # 实际可执行的训练、评估、汇总入口
docs/REPRODUCTION_REPORT.md          # 修改映射、保留机制、例外、验证和命令
artifacts/shared_protocol/           # 可导出的共同实验清单
  task_manifest.json 或 task_manifest.npz
  dataset_manifest.json
  observation_normalizer.npz
runs/<method>/<environment>/seed<k>/
  resolved_config.json
  provenance.json
  progress.csv
  checkpoints/
  evaluation/given_context/
  evaluation/self_collected/
```

交接报告应包含：

1. 原代码来源和 commit、补丁范围、保留的原生机制。
2. 公共配置映射和所有例外；方法特有超参数及出处；实际 actor/critic/encoder/辅助更新频率。
3. 任务/数据/normalizer 指纹及运行环境版本；若工件未生成则标明原因。
4. 已执行验证的命令、结果和边界；未执行的完整训练或真实环境验证。
5. 六环境 × 四 seed 的启动命令、断点恢复方式、两种主评估、最终 checkpoint 汇总命令。命令与目标 CLI 对应，不假设所有方法都叫 `train_gentle.py`。
6. checkpoint、日志及结果路径；每个运行的成功/失败/未运行状态。
7. 原版算法、共享协议和额外诊断的命名区分，避免后续写论文时误用结果。

主实验共六个基线加本文方法，每方法 6 环境 × 4 seeds，共 **168 个完整训练运行**，不包含预训练、消融和额外任务划分。每个模型可产生两种主评估结果，评估协议数量不等于再训练一套模型。尚未执行的运行应标为计划，不得预填回报。

## 7. 最小结果格式与来源记录

逐轨迹结果至少保存：

```text
method, environment, protocol_id, training_seed, checkpoint_iteration,
task_id, split, episode_index, episode_return_raw, episode_length,
context_source, context_pool_size, encoded_context_size,
initial_collection_steps, cumulative_evaluation_steps,
evaluation_seed_manifest_hash, collector_seed, context_seed, posterior_seed, rollout_seed
```

诊断结果额外保存 `context_repeat`、`reset_seed`、`context_bank_hash`、`collector_id`。轨迹文件或独立数组保存实际采样行 ID；避免只保存一个不可复核的平均数。

运行级来源记录至少保存：目标源码 commit/补丁指纹、参考协议版本、完整配置、任务/数据/normalizer 指纹、模型 SHA-256、依赖/模拟器版本、硬件、训练开始/结束时间、实际主更新数、各 optimizer 更新数、预训练成本、失败状态。数据校验和应从实际读取文件生成，不能仅对路径字符串取哈希。

论文主表汇总输出建议每行包含：`method, environment, protocol_id, checkpoint, n_training_seeds, mean_return, sample_std`，并保留各 seed 分数供复查。任何额外运行条件或例外需要能够从结果追溯到报告。

## 8. 当前已知限制与完成标准

本次文档整理时，参考工作区未发现本地 `data/`、`logs/`、`encoder_decoder/` 或 `dynamics/` 工件；根配置中的数据路径指向服务器。因此，本文核对的是源代码和配置，没有验证服务器文件数量、数据质量或训练结果。目标机器实际是否具备数据和依赖需要在执行时检查。

最终验收清单：

- [ ] 已取得并标识本次基线的原始实现，方法名称与实际变体一致。
- [ ] 六环境的任务参数及划分与共同清单一致，没有误用 placeholder。
- [ ] 离线文件、终止规则、归一化和动作/奖励单位一致，测试数据不进入训练。
- [ ] 公共参数实际生效，原生算法机制保留，例外有原因和量化记录。
- [ ] 200000 次主更新与各辅助更新可核对，训练 seeds 为 0/1/2/3。
- [ ] 两种主评估符合预算和累计 context 规则，native collector 的差异有说明。
- [ ] 最终 checkpoint 499 的选择与均值/样本标准差汇总正确。
- [ ] 固定 context 诊断独立命名，若执行则所有方法复用同一 bank。
- [ ] 代码、配置、报告、实际命令与验证记录齐备；未完成项明确列出。
- [ ] 正式结果仅来自完整真实运行，未将 smoke、模拟数据或缺失 seed 当成正式结果。

## 9. 参考代码导航与论文入口

下列路径均相对本文开头的参考仓库根目录，用于追溯协议，不要求目标仓库沿用同名文件。

| 核对内容 | 参考位置 |
| --- | --- |
| 默认参数与六环境覆盖 | `configs/default.py`；`configs/{point-robot,ant-dir,cheetah-vel,cheetah-dir,hopper-rand-params,walker-rand-params}.json` |
| CLI seed、网络、任务划分、normalizer 保存 | `train_gentle.py` |
| 数据导入、terminal、训练循环、两种主评估 | `rlkit/core/rl_algorithm.py` |
| context 抽样与 rollout | `rlkit/samplers/util.py`、`rlkit/samplers/in_place.py`、`rlkit/data_management/simple_replay_buffer.py` |
| task posterior 和策略动作 | `rlkit/torch/sac/agent.py`、`rlkit/torch/sac/policies.py` |
| 归一化与动作映射 | `rlkit/envs/wrappers.py`、`rlkit/torch/pytorch_util.py` |
| 环境选择与任务定义 | `rlkit/envs/__init__.py`、对应环境文件、`rand_param_envs/base.py` |
| 数据生成 | `policy_train.py`、`policy_eval.py`、`rlkit/torch/sac/pytorch_sac/train.py` |
| 本文方法及专用预训练 | `rlkit/torch/algo/gentle.py`、`pretrain_dynamics.py`、`pretrain_encoder_decoder.py` |
| 固定 context 诊断 | `evaluate_fixed_context.py`、`evaluate_fixed_context_suite.py`、`rlkit/torch/fixed_context_eval.py`、`docs/fixed_context_diagnostics.md` |

方法依据优先使用论文及其链接的作者代码，记录实际下载版本；不要把社区复现默认当作官方版本：

- [FOCAL（Li et al., 2021）](https://openreview.net/pdf?id=8cpHIfgY4Dj)
- [CORRO（Yuan and Lu, 2022）](https://proceedings.mlr.press/v162/yuan22a.html)
- [CSRO（Gao et al., 2023）](https://proceedings.neurips.cc/paper_files/paper/2023/file/fd489a44f3bcb9f122e4931ef21d0c43-Paper-Conference.pdf)
- [GENTLE（Zhou et al., 2024）](https://ojs.aaai.org/index.php/AAAI/article/download/29658/31121)
- [UNICORN（Li et al., 2024；本实验使用 SS 变体）](https://proceedings.neurips.cc/paper_files/paper/2024/file/8a30aba6514b56d02976f49797f6338a-Paper-Conference.pdf)；[作者代码](https://github.com/betray12138/UNICORN)
- [ER-TRL（Nakhaeinezhadfard et al., 2025）](https://ojs.aaai.org/index.php/AAAI/article/view/34160)；[作者代码](https://github.com/MohammadrezaNakhaei/ER-TRL)
- [深度强化学习实验统计参考（Agarwal et al., 2021）](https://arxiv.org/abs/2108.13264)

后续若修改共享协议，应更新版本号并列出变化及受影响的运行；不能将不同任务、数据或评估定义的结果放入同一主表而不注明。
