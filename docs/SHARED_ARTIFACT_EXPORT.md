# 共享任务与归一化文件：导出和接入

日期：2026-10-11。协议：`shared-main-v1`。对应需求：FOCAL 仓库中的 `docs/GENTLE_SHARED_ARTIFACT_EXPORT_HANDOFF.md`。

本次已实现独立导出入口和 GENTLE 消费入口。导出不需要模型权重，不启动训练。**本地没有服务器原始数据，六个环境的正式共享包尚未生成。** 本地测试中的合成轨迹均位于独立测试目录，不能用于正式比较。

## 1. 交付内容

- `export_shared_artifacts.py`：逐环境验证并发布共享包；`--help` 不需要 NumPy、Torch 或 MuJoCo。
- `shared_protocol/`：FOCAL 兼容的严格 schema、源文件锁、数据验证、任务加载和 GENTLE 接入。
- GENTLE 的 `pretrain_dynamics.py`、`pretrain_encoder_decoder.py`、`train_gentle.py` 新增 `--shared-artifacts` 和 `--data-dir`。
- `evaluate_fixed_context.py` 自动读取共享模式运行目录中保存的同字节工件；支持迁移后的共享包路径覆盖。
- `tests/test_shared_artifacts.py`：数据隔离、篡改拒绝、不可重拟合、预训练/评估接入以及实际 FOCAL 消费端兼容测试。

兼容层依据本地 FOCAL 的 `focal_shared/{common,data,environments}.py` 和 `reference_sources.json` 移植，运行时不依赖 FOCAL 仓库，也不会将两个仓库的 `rlkit` 混在同一个进程。归一化计算复用 GENTLE 的 `compute_observation_stats`。环境源文件未改动，沿用 FOCAL 锁定的环境 Python/XML 指纹和参考 commit。

## 2. 服务器导出命令

先同步本次新增目录、入口及修改的训练/评估文件到服务器 GENTLE 工作区。保留参考环境文件，不修改源码锁来绕过校验。

```bash
export REF_ROOT="$HOME/tievnas2/YinCH/GENTLE-consistant"
export FOCAL_ROOT="$HOME/tievnas2/YinCH/FOCAL"
export SHARED_ROOT="$HOME/tievnas2/YinCH/shared_protocol/shared-main-v1"
cd "$REF_ROOT"
python export_shared_artifacts.py --help
```

激活真实环境对应的 Python 环境：`gym==0.17.2`；Point 无需 MuJoCo；Ant/Cheetah 使用参考 `mujoco_py` 后端；Hopper/Walker 使用参考内置 MuJoCo 1.31 后端及 XML。具体安装位置由服务器已有环境决定。工具直接构造真实类并逐任务 reset/step，缺少后端会失败，不回退到 `non_mujoco_*`。

确认数据由参考采集代码保存为归一化策略动作后，定义：

```bash
export_env() {
    local env_name="$1"
    python export_shared_artifacts.py \
        --env "$env_name" \
        --data-dir "$REF_ROOT/data/$env_name" \
        --output-root "$SHARED_ROOT" \
        --data-actions normalized \
        --action-evidence "Reference policy_eval.py / Workspace.collect; stored policy-coordinate actions before NormalizedBoxEnv mapping"
}
```

`--action-evidence` 是实际采集来源说明，不是工具自动证明了历史采集方式。若服务器曾改过采集器，先核对；实际保存物理动作时使用 `--data-actions physical` 并写明证据。工具会检查动作范围，但数值范围本身不能证明坐标系。

在各自正确后端中逐条执行，检查每条命令退出状态：

```bash
export_env point-robot
export_env ant-dir
export_env cheetah-vel
export_env cheetah-dir
export_env hopper-rand-params
export_env walker-rand-params
```

每个环境只导出一次，四个训练 seed 和全部方法共用。数据根目录是单环境目录，例如 `.../data/point-robot`，不是其父目录。

如果以后需要复用已经核验过的已有文件，可增加以下选项；先发布过的包更适合直接复用整包目录：

```bash
python export_shared_artifacts.py \
    --env point-robot --data-dir "$REF_ROOT/data/point-robot" \
    --output-root "$HOME/tievnas2/YinCH/shared_protocol/imported-v1" \
    --data-actions normalized --action-evidence "Verified original collection code" \
    --normalizer /absolute/path/observation_normalizer.npz \
    --task-manifest /absolute/path/task_manifest.json
```

已有文件通过统计和任务校验后逐字节复制，不重写为“近似相同”的新文件。

## 3. 输出与失败处理

```text
<SHARED_ROOT>/
  point-robot/
    task_manifest.json
    observation_normalizer.npz
    dataset_manifest.json
    preparation_provenance.json
    validation.json
  ...其他环境...
  export_report.md
```

三个核心文件的内容与 FOCAL 消费接口一致；时间、后端、源码、dtype、历史证据等额外信息单独存放在 provenance/validation 中。

工具要求每个指定任务/轨迹文件存在，按任务 ID、轨迹编号、原始行序读取。normalizer 仅从训练 current observations 一次计算，使用原始 dtype 的归约语义、初始 count=1e-4 和总体方差。新导出的 NPZ 包含标量 count、format_version=1 及训练/测试 ID；Cheetah-Dir 的同一任务不会在统计中重复计算。

完整包先在临时目录构建并验证，再整体发布。已存在的完整包重新执行时验证后复用，不覆盖；不完整目录直接报错。失败不会把 staging 目录发布为正式包，失败原因写入 `<env>.failure.json`。遗留的 `.环境名-staging-*` 是失败诊断文件，不应提供给训练使用。进程被强制终止可能留下 `.环境名.export-lock`，确认没有导出进程后再处理该锁，或选择新输出根目录。

`export_report.md` 汇总六环境实际状态、路径、文件数、实测行数和核心哈希。不存在实际数据的环境会显示未执行或失败，不生成猜测数组。

若存在 `goal_idx*/log.npy`，工具检查可识别的 task/goal 或物理参数记录，冲突时报错。缺少记录或格式无法识别时，provenance 明确记录历史任务一致性未完全验证。工具根据锁定源码重建的任务清单通过，并不等于已证明历史数据从未改变。

## 4. FOCAL 接入与交叉校验

正式运行可直接将 FOCAL 的 `--artifacts` 指向 `$SHARED_ROOT/<env>`，无需重新 prepare。以下 prepare 用于独立交叉验证，输出到新的验证目录：

```bash
cd "$FOCAL_ROOT"
python scripts/shared_protocol.py prepare \
    --env point-robot --reference-root "$REF_ROOT" \
    --data-dir "$REF_ROOT/data/point-robot" \
    --artifacts "$FOCAL_ROOT/artifacts/shared_protocol/import-check/point-robot" \
    --data-actions normalized \
    --normalizer "$SHARED_ROOT/point-robot/observation_normalizer.npz" \
    --task-manifest "$SHARED_ROOT/point-robot/task_manifest.json"
```

换成其他环境名可逐个验证；已有验证输出时使用新目录。核对 task JSON 和 NPZ 文件 SHA-256 完全相同，dataset JSON 的 canonical digest 相同。dataset JSON 因序列化格式不同可能字节不同，因此正式实验优先直接引用已发布的完整包。

真实数据短验证（按实际设备修改，输出独立于正式训练）：

```bash
python scripts/shared_protocol.py smoke \
    --env point-robot --reference-root "$REF_ROOT" \
    --data-dir "$REF_ROOT/data/point-robot" \
    --artifacts "$SHARED_ROOT/point-robot" \
    --runs-root "$FOCAL_ROOT/runs-shared-artifact-check" \
    --seeds 0 --smoke-updates 2 --device cpu
```

FOCAL 的训练命令只需沿用相同 `--artifacts` 和原始 `--data-dir`。这些服务器交叉验证命令尚未在真实服务器数据上执行，不属于已经完成的结果。

## 5. GENTLE 接入

三个训练入口均使用相同的两个路径参数。以下是 Point、seed 0 的命令；其他环境替换配置名和共享包目录，其他 seed 替换 `--seed_list`。这部分是后续训练接入说明，**导出文件不需要执行这些训练命令**。

```bash
cd "$REF_ROOT"
python pretrain_dynamics.py configs/point-robot.json \
    --seed_list 0 --gpu 0 \
    --data-dir "$REF_ROOT/data/point-robot" \
    --shared-artifacts "$SHARED_ROOT/point-robot"

python pretrain_encoder_decoder.py configs/point-robot.json \
    --seed_list 0 --gpu 0 \
    --data-dir "$REF_ROOT/data/point-robot" \
    --shared-artifacts "$SHARED_ROOT/point-robot"

python train_gentle.py configs/point-robot.json \
    --seed_list 0 --gpu 0 --exp_name shared_v1 \
    --data-dir "$REF_ROOT/data/point-robot" \
    --shared-artifacts "$SHARED_ROOT/point-robot"
```

共享模式在启动时校验全部任务、数据文件和统计，然后安装精确任务数组并 reset 当前任务。环境和离线 replay 使用同一变换，进入 buffer 时只归一化一次，不再调用统计 update。算法目标、网络和正式训练预算未修改；未提供共享选项时保留旧运行分支。

共享模式的预训练产物自动放在：

```text
dynamics/<env>/shared-<dataset_manifest文件SHA前16位>/expert_seed<seed>/
encoder_decoder/<env>/shared-<同一SHA前16位>/expert_seed<seed>/
```

主训练从相同目录加载，并要求其保存的三个核心文件哈希一致。这样不会把旧 `expert_seed*` 权重直接绑定到新统计。共享预训练目录已存在时会拒绝覆盖；失败重试前先检查该目录，不能把它当作完整预训练结果。显式 `--path_to_weights` 也必须通过共享工件校验。

主训练日志、预训练输出中保存核心文件的同字节副本及 `shared_artifact_usage.json`，variant 中记录共享路径和三文件 SHA-256。旧 checkpoint 没有这套来源记录时，不能靠新增参数就宣称符合共享协议。

固定 context 评估从运行目录读取同字节副本；迁移数据时传原有 `--data-dir`，迁移工件时可传 `--shared-artifacts`：

```bash
python evaluate_fixed_context.py \
    --log-dir /absolute/path/to/shared-mode-run \
    --epochs 499 --offline-only \
    --data-dir "$REF_ROOT/data/point-robot" \
    --shared-artifacts "$SHARED_ROOT/point-robot" \
    --output-dir /absolute/path/to/separate-diagnostic-output
```

没有共享模式来源记录的历史模型会拒绝这个覆盖选项；不修改旧模型原始统计。主训练内两种评估自然使用已经安装共享统计与任务的同一个环境。

## 6. 验证范围与服务器待办

本地使用已有 Python 3.11、NumPy 1.26.4、Gym 0.17.2、Torch 2.2.1 环境验证；这不是声称服务器运行栈相同。真实 Point 环境已构造并 reset/step。MuJoCo 五环境未在本机运行。

首批相关测试 107 项通过，包括实际 FOCAL `prepare` 和 `load_prepared` 对独立测试包的读取、相同字节复用、数据清单语义一致，以及任务/统计/原始文件篡改拒绝。随后新增链路测试并运行针对性回归，26 项通过，覆盖两步真实 encoder–decoder 优化、权重保存/加载和共享模式固定 context 评估。Dynamics 入口使用记录器验证送入训练的数据与共享统计一致，不把该记录器测试称为 dynamics 模型训练完成。测试使用 2000 个临时合成 Point 轨迹文件，仅用于代码验证。

独立导出 CLI 也已在隔离的合成测试数据上完整执行，生成五文件包和总报告；三个训练入口及评估入口的 `--help` 已实际检查。GENTLE 训练入口的导入验证使用临时目录中安装的 `gtimer`，未修改系统或 FOCAL 的 Python 环境。相关测试产物位于本仓库 `output/shared-artifact-tests/`，名称含 TEST_ONLY/TEST-ONLY 的数据不能发布为实验共享文件。

可复跑测试：

```bash
python -m pytest tests/test_shared_artifacts.py tests/test_fixed_context_data.py \
    tests/test_fixed_context_cli.py tests/test_interpolation_ablation.py \
    tests/test_training_logging.py -q
```

跨仓库测试默认查找并列的 `FOCAL` 目录；也可设置 `FOCAL_SHARED_ROOT`。缺少该消费端时明确 skip，其余测试仍运行。Windows 沙箱中默认临时目录若不允许原子重命名，使用工作区内一个**尚不存在**的 `--basetemp`，并先创建其父目录；不要为测试删除现有实验目录。

| 环境 | 本地真实后端检查 | 正式数据包导出 | 真实数据消费者验证 |
| --- | --- | --- | --- |
| Point-Robot | 已通过 | 待服务器数据 | 待服务器运行 |
| Ant-Dir | 待 MuJoCo | 待服务器数据 | 待服务器运行 |
| Cheetah-Vel | 待 MuJoCo | 待服务器数据 | 待服务器运行 |
| Cheetah-Dir | 待 MuJoCo | 待服务器数据 | 待服务器运行 |
| Hopper-Rand-Params | 待 MuJoCo 1.31 | 待服务器数据 | 待服务器运行 |
| Walker-Rand-Params | 待 MuJoCo 1.31 | 待服务器数据 | 待服务器运行 |

服务器执行后，以实际生成的 `export_report.md` 和消费者日志更新上述状态；不要把本地接口测试填成正式数据导出成功。本次未重新收集数据、未启动完整训练，也不生成 fixed-context bank 作为共享包内容。
