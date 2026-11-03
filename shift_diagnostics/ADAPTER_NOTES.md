# 原生接口与证据边界（2026-11-02，v3）

依据：`A-evidence-20261102T121715Z-5124/evidence.zip`，全部 2,203 个成员哈希通过。当前本地/服务器报告 commit=`49857cc2bc0f7d343b1695c7e645497152982459`；行为以实际源码哈希为准。初次笔记归档于 `diagnostics_audits/ADAPTER_NOTES-initial-20261102.md`。

## 原生接口

| 实现 | 编码与动作 | 关键校验 |
| --- | --- | --- |
| GENTLE/Ours | autoencoder.MlpEncoder + Agent；原生确定性 actor | 同 run/iteration 分离权重 strict load。Cheetah latent5、s/a/r；Walker latent10、s/a/r/s'；原生训练 RMS |
| FOCAL | [200,200,200] encoder + PEARLAgent | iteration499；binding、训练/优化器计数、LF 源码哈希；raw obs |
| CORRO | normalize_z 的 MLPEncoder + 训练好的 SelfAttnEncoder | generative 完整版本、iteration999、config/artifact/source、policy_updates；raw obs |
| CSRO | 原生 MlpEncoder/PEARLAgent，保留 layer_norm 配置 | iteration499、非 test_only、seed/scientific_config/source/artifacts；shared RMS |
| UNICORN-SS | Gaussian transition latent→tanh→mean | SS、checkpoint iteration1000、seed/config/source/artifacts/main_updates；raw obs，原生 episode-end 输入 |
| ER-TRL | 原生 encoder_dims + rlkit.torch.agents.PEARLAgent | iteration99、完整训练预算、非 smoke、LF 源码哈希和共享文件绑定；raw obs |

只构建推理网络，不构建训练算法；eval、requires_grad=False、no_grad。每方法单独进程，避免同名 rlkit/models/torchkit 污染。前后校验模型参数和归一化统计哈希。

UNICORN 表征计算 E[tanh(Z)]，使用 128/256 点 Gauss-Hermite 积分交叉检查误差≤1e-4；不使用 tanh(E[Z]) 替代。控制 latent 仍调用原生随机 context_encoding，隔离 posterior RNG；用两个不同 RNG 验证表征稳定。

## 数据和环境

- 每环境唯一 GENTLE seed0/499、z=0、确定性 actor 生成 smoke raw bank，其他方法只读。数组内容、dtype、shape 都参与哈希。
- raw obs/s' 在模型适配器内做一次原生变换；环境不另包观测 RMS。normalized action 到物理 action 映射一次，reward 不乘训练 Bellman scale。
- 配对 reset seed 排除 method/training seed；posterior RNG 单独隔离。rollout 固定 context，不吸收反馈。
- Gym0.17.2；Cheetah MuJoCo1.50、Walker1.31，独立进程，启动前设置 LD_LIBRARY_PATH；核实实际动态库及 XML。
- manifest 直接向环境构造提供任务参数/顺序，不重新抽样；核验 reference source。
- 四列离线数据无真实 absorbing-terminal 信息；原生末行 boundary 语义单独记录，UNICORN 保留其原生输入语义。

## GENTLE/Ours 历史证据限制

所选训练流程未保存独立 RMS。只使用 hash-verified 原始 seen 训练轨迹，按 rl_algorithm.init_buffer 的 trajectory-major glob 顺序、原始 dtype 做一次 RunningMeanStd 更新；记录文件顺序、RMS 哈希、与 published RMS 的差异。不在 context/unseen 上拟合。

历史 glob 枚举顺序未存档，不能证明重建值逐 bit 等于训练时 RMS。历史 state_dict 没有内嵌 seed/source 签名；用户指定同 run 配对和外部证据不能被描述成 checkpoint 自带身份认证。

本地测试包含六套实现×两个环境配置的接口检查，使用随机 fixture 权重。真实 checkpoint 与 MuJoCo 尚待服务器执行 run_stage_a.py 验收，A 仍未完成。
