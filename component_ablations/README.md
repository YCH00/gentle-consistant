# 三组件消融：运行说明

这是独立的 `component-ablations-v1` 实验，六臂复用一套训练矩阵。它不调用 FOCAL/CORRO/CSRO 等完整方法比较，也不改写 `shift-diagnostics` 产物。

## 已锁定的定义

用户在 2026-11-03 明确确认：**P 包括虚拟 critic、actor-Q 和 actor-BC**。这覆盖交接文档中“BC 仅用于真实数据”的默认说明。

- R/V 使用开关之前的真实、虚拟元素总数确定各自有效系数；禁用一项不会改变另一项的系数。
- Critic、actor-Q、actor-BC 分别保留真实分支均值，加上 `P * w(t) * 虚拟分支均值`。Actor 的 alpha 只由真实 Q 计算。
- 虚拟权重使用原完整配置的 decay；recon/cycle 权重、距离日志关闭。恒定权重仅用于编号 47 的补充对照。
- P=0 仍执行虚拟生成和辅助采样，然后屏蔽优化贡献。六臂的真实更新预算与采样随机流保持匹配。
- 新契约与历史混合分母实现不同，48 个核心槽位全部重新训练。`backbone` 名为 Matched backbone；`full` 名为 Matched full。

| 默认编号 | arm | R | V | P | 作用 |
|---|---|---:|---:|---:|---|
| 40 | backbone | 0 | 0 | 0 | 匹配基础框架 |
| 41 | real_only | 1 | 0 | 0 | 真实一致性独立贡献 |
| 42 | no_real | 0 | 1 | 1 | 去除 R |
| 43 | no_virtual_recovery | 1 | 0 | 1 | 去除 V |
| 44 | no_virtual_rl | 1 | 1 | 0 | 去除 P，同时复用为零权重对照 |
| 45 | full | 1 | 1 | 1 | 完整 Local + decay |
| 46 | full_global | 1 | 1 | 1 | Global 生成补充对照 |
| 47 | full_constant | 1 | 1 | 1 | 恒定初始权重补充对照 |

每个编号包含 cheetah-vel、walker-rand-params 两环境与 0/1/2/3 四种子。核心 48 次；加上两个补充配置共 64 次。编号按配置分配，不按种子重复编号。

Local 基准取自已记录的 `23_local_decay` resolved configs，不从当前 default 猜测。Walker 为 Local、`n_vt=10`；Cheetah 的 `M=1`。Global 使用历史 `28_global_decay` 的生成参数 `M=2`，因为 Global 的 `M=1` 会退化成单个真实任务；该差异已登记，不当成损失差异。

两环境均为 500 iterations × 400 主更新。当前原生代码虽然配置了 `policy_freq=2`，实际每一步都更新 actor 和 targets；本实现保持这个真实行为，并记录 200,000 次 actor、critic 更新和 200,000 次 target 更新事件（每事件三个网络）。

## 服务器一键运行已准备好的计划

将新增的 `component_ablations/`、`tests/test_component*.py`、`run_component_ablations.py`、`component-ablations-v1/` 放到对应服务器仓库内；使用原来的训练 conda 环境。下面的命令从仓库根目录执行：

```bash
conda activate gentle
bash component-ablations-v1/commands.sh component-ablations-v1 0,1,2,3 1
```

参数依次为项目目录、物理 GPU 列表、每卡进程数。默认每卡一个训练进程，自动先完成 48 个核心槽位，再执行补充训练。某个任务失败会保留日志并继续其他可执行任务；阶段命令返回非零，整条脚本停止在相应阶段。

父进程在导入 torch 之前给每个子进程设置 `CUDA_VISIBLE_DEVICES`。Cheetah 使用 MuJoCo 150，Walker 使用 131；两种后端可以在不同子进程同时运行。默认引擎目录 `~/.mujoco/mjpro150` 和 `~/.mujoco/mjpro131`，原有 license/driver 路径沿用当前环境。

## 创建自己的实验名称与编号

```bash
python -m component_ablations plan \
  --root component-ablations-exp44 \
  --start-id 40 --name exp44 --suite all \
  --gpus 0,1,2,3 --jobs-per-gpu 1 \
  --evidence /root/tievnas2/YinCH/shift-diagnostics-v3/smoke/A-20261103T011150Z-15243 \
  --data-root /root/tievnas2/YinCH/GENTLE-consistant/data \
  --weights-root /root/tievnas2/YinCH/GENTLE-consistant/encoder_decoder \
  --dynamics-root /root/tievnas2/YinCH/GENTLE-consistant/dynamics
```

`--suite core` 只计划六臂；`all` 计划八配置。名称成为 `40_exp44_backbone` 等，并用于 `runs/<environment>/<experiment_name>/seed<k>/`。已存在的协议不能被 `plan` 覆盖。仅改变 GPU 或队列并行度不用重建计划。

`--evidence` 只读取历史 resolved config、任务/数据哈希与训练 normalizer 的来源记录；其中的 smoke context bank 不作为正式评估 bank。已准备好的项目运行时无需这个旧证据目录。

项目内的 `planning_evidence/` 保存配置测试需要的 10 个审计输入文件，包含两个环境的 `banks/<env>/manifest.json` 后端记录；它不包含评估 context bank。默认旧证据目录不存在时，程序自动读取 `component-ablations-v1/planning_evidence/`。测试直接使用这份随项目携带的资料，并检查新计划仍携带完整资料。

## 按阶段执行

```bash
# A：核验固定数据哈希，首次登记实际预训练文件哈希。
python -m component_ablations audit --root component-ablations-v1 --seal-pretrained

# B：配置差异白名单 + 数值/梯度/完整状态恢复/队列测试。
python -m component_ablations check --root component-ablations-v1

# 仅预览命令；不会创建训练输出，也不会绕过正式运行的阶段检查。
python -m component_ablations launch --root component-ablations-v1 \
  --phase train --gpus 2,3 --jobs-per-gpu 1 --max-parallel 2 --dry-run

# 同一环境、同一种子先创建一份不可变初始化，六臂读取同一文件。
python -m component_ablations launch --root component-ablations-v1 --phase init --gpus 2,3

# 默认 seed0；两环境六核心臂共 12 个真实环境 smoke；all 计划再加 4 个补充 smoke。
python -m component_ablations launch --root component-ablations-v1 --phase smoke --gpus 2,3

# C：先核心，再补充；缺少真实 smoke 或资源会失败。
python -m component_ablations launch --root component-ablations-v1 --phase train --gpus 2,3
```

可通过 `--envs cheetah-vel`、`--arms full no_virtual_rl`、`--seeds 0 1` 选择子集；比较表仍要求四种子完整。自定义后端位置用 `--mujoco-root /path/to/.mujoco`。Python 解释器用 `--python /path/to/gentle/bin/python`。

审计发现缺失文件或哈希不符时，先修正 `resources.json` 中的实际路径，再运行 audit。`--seal-pretrained` 只为尚未登记的预训练文件设置哈希，不会接受已登记文件的哈希变化。开始训练后应冻结资源和源码；变更损失契约需创建新项目。缺失 encoder/decoder/dynamics 不允许回退到随机初始化。

## 失败与完整恢复

```bash
python -m component_ablations launch --root component-ablations-v1 \
  --phase train --gpus 2,3 --arms full --seeds 2 --resume
```

`last_checkpoint.json` 指向两个轮换的完整恢复文件之一，保存 optimizer、targets、随机状态、可变 replay/relabel/virtual buffers、计数器、schedule/EMA 与统计累计值。每个 iteration 原子提交；中途失败恢复到上一个已提交 iteration，未提交的日志被截去。固定科学检查点 0/200/400/499 不覆盖；重放若与已存在的检查点参数不一致会报错。

原生仅网络权重的 checkpoint 不能用来精确续训。若第一轮还没有完整恢复点，则不能冒充 resume，需保留失败目录并建立新的重启记录/项目。模型、配置、初始化和资源哈希不一致也拒绝恢复。

## 主评估、固定 context 与 probe

训练在 iteration 0/200/400/499 自动保存科学检查点并运行 Given-context / One-shot 主评估。每任务先建立固定池，One-shot 恰好采集该 arm 自身 policy 的 200 条真实 transition；每条评估轨迹前有放回抽 200 条并累计 context，评估轨迹不回流，完整 episode 累积至少 600 步。初始采集回报不计入评估。各 reset/context 种子预先固定并跨 arm 配对。

固定 context 和 probe 要先绑定**正式且完整**的资源，不能拿已有 v3 smoke 目录替代：

```bash
python -m component_ablations bind-diagnostics --root component-ablations-v1 \
  --env cheetah-vel --bank /path/to/formal/cheetah/manifest.json \
  --probe /path/to/formal/cheetah/probe_manifest.json
python -m component_ablations bind-diagnostics --root component-ablations-v1 \
  --env walker-rand-params --bank /path/to/formal/walker/manifest.json \
  --probe /path/to/formal/walker/probe_manifest.json
python -m component_ablations audit --root component-ablations-v1
python -m component_ablations launch --root component-ablations-v1 --phase fixed --gpus 2,3
python -m component_ablations launch --root component-ablations-v1 --phase probe --gpus 2,3
python -m component_ablations summarize --root component-ablations-v1
python -m component_ablations plot --root component-ablations-v1
```

接口格式见 [RESOURCE_SCHEMAS.md](RESOURCE_SCHEMAS.md)。当前本地尚无正式 bank/probe 数据，因此这些接口未通过真实资源端到端验收，阶段 D 保持待执行。

固定评估使用 offline、policy499_sigma000、policy499_sigma020 三条件，每环境 20 个任务，每条件/任务 20 个长度 200 的 context，每个 context 三个固定 reset。Probe 冻结 encoder，seen train 训练、seen val 选模型，unseen 只测试；两层 128 ReLU、Adam 1e-3、batch256、10,000 updates，每500步验证。拒绝 context/query 或 train/val/test 轨迹重叠。

## 产物与验收边界

- `STATUS.md`：阶段、命令、验证与资源阻塞；`resource_audit.json`：每文件核验结果。
- `loss_contract.json` / `protocol.json`：损失、梯度路径、预算、生成器、BC 决策与配对关系。
- `run_matrix.json`：每个槽位的状态、名称、配置、初始/最终模型与数据哈希。
- `job_logs/`、`jobs_<phase>.json`、`failures/`：子进程命令、GPU、退出码、错误。
- `runs/.../training_steps.jsonl`：逐主更新 raw/weighted R/V、Q/actor/BC、计划/有效权重、虚拟使用量。
- `runs/.../progress.csv`、`completion.json`：预算、冻结 decoder 校验、初始化哈希、时间与显存。历史预训练耗时未知时为 null，不填写估计值。
- `summaries/`：主表、五个预定义配对差值、R 噪声回报变化/几何指标、V probe 误差、P 使用量与成本；JSON/CSV/Markdown/LaTeX。
- `figures/`：有完整真实结果后才生成 PDF/SVG/PNG 和可复算输入。

统计单位始终为四个训练 seed，std 使用 `ddof=1`。缺种子时 mean/std 为 null，表格显示 incomplete；零资源时不会生成模拟回报或性能图。局部数值测试通过不等于两环境真实 smoke 或正式训练完成。
