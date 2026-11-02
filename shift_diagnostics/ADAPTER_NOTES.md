# 阶段 A 源码核验笔记（2026-11-02）

范围：当前本地 checkout `6e44eecb5427e57a8ac3453858d53f3f418f6908`。这是静态源码证据，不代表服务器 checkpoint 已通过适配或已确认本仓库对应原版 GENTLE/Ours。

## 当前 GENTLE 风格接口

| 项目 | 本地证据 | 适配要求 |
| --- | --- | --- |
| 网络构造 | `train_gentle.py` 构造三层 MLP encoder/actor，encoder output_activation=tanh | 必须按实际 run 配置和对应版本构造，不能从本地默认配置猜正式网络 |
| 字段 | `Agent.update_context_dict` 拼接 obs/action/reward，按 use_next_obs_in_context 加 next_obs | 接收同一 raw bank，状态转换一次，保留原生字段，reward 不乘 Bellman scale |
| 确定性 latent | `Agent.infer_posterior` 无 IB 时对 transition 编码均值，sample_z 取 z_means | 用原生均值作为表示；z_vars 在此分支记录的是 std，不能误认为高斯 posterior variance |
| 概率 latent | IB 时 softplus 方差 + product of Gaussians，infer_posterior 内会 sample_z | 表征读 z_means；控制保留原生采样。隔离采样 RNG，不能通过强制均值动作一并取消 posterior 随机性 |
| 状态清理 | `Agent.clear_z` 清空 context 并重置 prior，也会 sample_z | 每 context/rollout 清理；prior 和 posterior RNG 不得影响环境 reset RNG |
| 动作 | `Agent.get_action(..., deterministic=True)`，policy.forward 的确定性分支返回 tanh(mean) | 采集器必须显式确定性，不能直接照搬 `ContextPolicyWrapper.get_action`，后者默认随机 actor |
| 保存 | `GENTLE.get_epoch_snapshot` 返回 policy/context_encoder 等 state_dict；logger 分别保存 | encoder/policy 同一 run、同一 iteration，分别哈希并 strict load；不从名称推断配对真实性 |
| 归一化 | `rl_algorithm.py` 从训练离线观测更新 RunningMeanStd，并调用 env.update_obs_mean_var | 该训练流程会拟合统计，本轮不能重跑它替代加载共享统计 |
| 归一化公式 | `RunningMeanStd.forward` 与 wrapper 均使用 `(obs-mean)/sqrt(var+1e-8)` | 最终以实际训练源码为准；保存文件字段/公式/精度未核验前不能套用通用 std 公式 |
| 环境包装 | `NormalizedBoxEnv` 将 [-1,1] 映射到物理动作；启用统计后 reset/step 返回归一化观测 | bank 必须保存 raw obs/next_obs。采集时控制观测归一化层；不能将 wrapper 输出误记为 raw 后再次归一化 |
| 环境注册 | `rlkit/envs/__init__.py` 按 MUJOCO_PY_MJPRO_PATH 选择真实后端；未设置时导入 non_mujoco | 必须核验真实环境模块、XML、后端与 reset/step；不能让无 MuJoCo 替身通过 smoke |
| 任务构造 | 环境构造函数调用 sample_tasks | 需核实服务器 shared runtime 的任务恢复实现，确保最终使用精确 manifest 参数及顺序，不把重新采样结果用作正式任务 |

当前 `train_gentle.experiment` 会额外加载 dynamics、创建训练算法并调用 train，不能直接用于只加载 encoder/policy 的诊断。

本地配置：Cheetah-Vel latent_size 继承默认 5，context 为 s/a/r；Walker 覆盖 latent_size=10，并启用 next_obs。两者默认无 information bottleneck。这些仅是本地配置事实，**不能填成服务器模型已确认属性**。

## 其余方法与版本

本地相邻目录发现 FOCAL、CORRO、CSRO、ER-TRL、UNICORN 及多个 GENTLE 版本，部分有 `run_shared_native_v2.py` 等入口；`FOCAL/focal_shared` 当前不存在。目录名称、入口名称、历史缓存都不能证明正式主实验版本。

接下来依据服务器报告定位各方法的 loader、配置和任务/normalizer 契约；尤其确认 UNICORN 为 SS 版本、GENTLE 为原版完整基线、Ours 为完整方法。没有实际证据的适配器保持 `not_verified`，不先写统一 MLP loader 假定七种方法结构相同。
