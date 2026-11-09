# 混合 R/M/E 数据训练 GENTLE

本改动只接入新的训练数据源，并保留当前仓库的算法配置和原生评测流程。
当前仓库含虚拟任务、consistency loss 等扩展，新增配置复制自各环境原配置；
不要将它自动视为关闭所有扩展后的原版 GENTLE 基线。

## 数据和评测

- 动力学、encoder/decoder 预训练、主训练统一按 v3 manifest 加载全部训练轨迹。
- 历史 `data/<env>` 继续为测试任务提供离线 context；Cheetah-Dir 的任务 ID 虽相同，数据池分开。
- 在线 context 沿用原 collector。评测预算、回报平均规则保持原设置，不接 frozen bank。
- 训练任务上的诊断指标仍使用训练池，因此其离线 context 会变成混合数据。
- 每个文件末行标为 terminal，保留真实长度。动作不重复缩放，奖励沿用原处理。
- 使用原 GENTLE RunningMeanStd 从混合训练当前观测拟合统计量，训练、历史测试轨迹、在线环境共享它。
  发布包中的 normalizer 会校验哈希，但不会替换原方法的计算规则。
- 保留旧配置和旧 `_step` 数据读取方式；新配置名为 `configs/<env>-mixed-rme-v3.json`。

## 服务器准备

用 Git 同步本次新增和修改的文件，尤其不要漏掉 loader、启动器、六个新 JSON 配置和测试文件。
在服务器进入仓库并启用原训练 Python 环境：

```bash
cd /root/tievnas2/YinCH/GENTLE-consistant
conda activate gentle
python -m unittest test_mixed_dataset -v
```

环境名以服务器实际已安装的为准。继续使用已有依赖版本，不需要升级训练依赖。
目录应为：

```text
/root/tievnas2/YinCH/
  GENTLE-consistant/
    data_mixed_rme_v3/<env>/goal_idx<id>/collection.json
    data_mixed_rme_v3/<env>/goal_idx<id>/trj_evalsample<k>.npy
    data/<env>/goal_idx<id>/trj_evalsample<k>_step<step>.npy
  shared_protocol/
    shared-native-v3/<env>/
      task_manifest.json
      dataset_manifest.json
      observation_normalizer.npz
      preparation_provenance.json
      validation.json
    shared-main-v1/<env>/
      task_manifest.json
      dataset_manifest.json
```

启动器从自身仓库目录定位 `data` 和 `data_mixed_rme_v3`，从仓库父目录定位 `shared_protocol`。
若共享工件在其他项目根目录，可加 `--project-root /实际项目根目录`。
校验历史 manifest 哈希和其引用的实际评测文件；不需要 frozen bank。

## 第一步：全环境数据校验

此步骤只需要 NumPy，不初始化 MuJoCo：

```bash
set -e
for env in point-robot ant-dir cheetah-vel cheetah-dir hopper-rand-params walker-rand-params; do
  python run_mixed_gentle.py validate --env "$env"
done
```

训练轨迹数应分别为 1000、500、500、200、500、500，总计 3200。
每任务行数以清单为准。任意文件、哈希、轨迹配额或任务划分不符会终止。
真实环境维度、动作边界及任务参数会在后续训练入口启动时再次核对。

## 第二步：真实环境 smoke

在服务器已有 `use_mujoco` 函数可用的 shell 中运行。Point-Robot 也使用 150 这组环境导入设置，
避免其他 MuJoCo 环境被无意替换成 `non_mujoco` 占位实现。

```bash
set -e
use_mujoco 150
for env in point-robot ant-dir cheetah-vel cheetah-dir; do
  python run_mixed_gentle.py smoke --env "$env" --gpu 0 --seed 0
done

use_mujoco 131
for env in hopper-rand-params walker-rand-params; do
  python run_mixed_gentle.py smoke --env "$env" --gpu 0 --seed 0
done
```

smoke 读取完整数据，动力学最多训练 1 epoch，encoder/decoder 训练 2 个外层迭代，
主训练 2 个外层迭代且每次 1 个梯度步骤。保留原评测预算并实际运行两种评测。
因此最后的评测和表示可视化仍可能花一些时间。
smoke 使用 `mixed_rme_v3_smoke` 权重和日志目录，不会覆盖正式权重。

## 第三步：正式训练

默认流程为：每个 seed 依次执行动力学预训练 → encoder/decoder 预训练 → 主训练。
默认 seeds 为 0、1、2、3，顺序执行，避免原入口默认多进程同时占用一张 GPU。

先运行一个环境：

```bash
use_mujoco 150
python run_mixed_gentle.py train --env ant-dir --gpu 0 --exp_name 1_ours
```

或者依次跑完六个环境：

```bash
set -e
use_mujoco 150
for env in point-robot ant-dir cheetah-vel cheetah-dir; do
  python run_mixed_gentle.py train --env "$env" --gpu 0 --exp_name 1_ours
done

use_mujoco 131
for env in hopper-rand-params walker-rand-params; do
  python run_mixed_gentle.py train --env "$env" --gpu 0 --exp_name 1_ours
done
```

单独执行指定种子：

```bash
python run_mixed_gentle.py train --env ant-dir --gpu 0 --exp_name 1_ours --seed 2
# 多个指定种子：--seed 2 --seed 3
```

启动器不自动跳过已有权重。重新执行 train 会重新进行所选 seed 的预训练和主训练，
覆盖该 seed 的预训练目录；主训练日志使用新的时间目录。它不是断点续训命令。

**encoder/decoder 预训练是否使用，要与旧实验保持一致。** 默认完整流程明确要求此阶段完成，
避免因旧目录中碰巧存在权重而隐式改变实验。
如果旧对照实验从随机 encoder/decoder 开始，则在 smoke 和 train 命令中都加：

```bash
--context-pretraining disabled
```

这会跳过 encoder/decoder 预训练，并禁止主训练自动加载该阶段权重；动力学仍会重训。
不要在一组对照结果里混用这两种训练设置。

如果旧实验使用不同算法超参数，可用 `--config /path/to/base.json` 提供完整基础配置。
启动器只覆盖混合/历史数据源、数据版本、预训练开关；正式训练不缩短优化步骤。

## 分阶段手动执行

固定服务器目录与新 JSON 配置一致时，可以手动运行同一个 seed 的三个阶段：

```bash
use_mujoco 150
python pretrain_dynamics.py configs/ant-dir-mixed-rme-v3.json --gpu 0 --seed_list 0
python pretrain_encoder_decoder.py configs/ant-dir-mixed-rme-v3.json --gpu 0 --seed_list 0
python train_gentle.py configs/ant-dir-mixed-rme-v3.json --gpu 0 --seed_list 0 --output_prefix mixed_rme_v3_ --exp_name 1_ours
```

每条命令都显式给出一个 `--seed_list`，否则旧入口会默认并发运行四个 seed。
三个阶段必须使用同一份配置和相同 seed。主训练检查预训练权重的数据身份及归一化统计量，
不允许混合训练集配旧 Expert 权重。

## 输出与读取结果

```text
dynamics/<env>/mixed_rme_v3_seed<seed>/
encoder_decoder/<env>/mixed_rme_v3_seed<seed>/
logs/<env>/mixed_rme_v3_gentle/seed<seed>/1_ours/
output/mixed_rme_v3/<env>/config.json
```

权重及主训练日志目录保存 `dataset_identity.json` 和实际使用的 `obs_normalizer.npz`。
主训练日志仍包含 `progress.csv`、TensorBoard 事件及模型权重。

- `AverageReturn_all_test_tasks`：历史离线 context 下的测试回报。
- `AverageReturn_all_test_tasks_expl`：原在线 context collector 下的测试回报。
- 正式结果使用原实验约定的模型选择规则；如果原规则是 final，则取最后迭代。
  当前默认 500 次外层迭代时，最后模型为 `*_itr_499.pth`，不要选 smoke 或按测试最高回报挑模型。
- 四个 seed 分别读取相同指标和对应迭代，再按原统计口径汇总。

本地验证包括合成数据测试和 Python 3.8 语法检查；真实服务器数据校验、MuJoCo smoke 与完整训练
需要在服务器执行，不能由本地合成测试替代。
