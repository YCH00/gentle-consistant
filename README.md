# Generalizable Task Representation Learning for Offline Meta-Reinforcement Learning with Data Limitations
Code for AAAI'24 paper "Generalizable Task Representation Learning for Offline Meta-Reinforcement Learning with Data Limitations".

## Installation

First install [MuJoCo](https://www.roboti.us/index.html). For tasks differ in reward functions (Cheetah, Ant), install MuJoCo150 or plus. Set `LD_LIBRARY_PATH` to point to both the MuJoCo binaries (`/$HOME/.mujoco/mujoco200/bin`) as well as the gpu drivers.

Then create conda environment by:

```bash
conda create -n gentle python=3.8
conda activate gentle
pip install -r requirements.txt
apt-get update
apt-get install -y libosmesa6-dev libosmesa6-dev libgl1-mesa-dev libgl1-mesa-glx libglew-dev patchelf gcc g++ libglfw3 libglfw3-dev
# apt-get install -y libosmesa6-dev libgl1-mesa-dev libgl1-mesa-glx libglew-dev patchelf
# apt-get install -y gcc g++ libglfw3 libglfw3-dev
python -m pip install "pip<24.1"
pip install hydra-core==1.0.7 omegaconf==2.0.6
pip install termcolor cffi lockfile
pip install --no-build-isolation mujoco-py==1.50.1.68
python -m pip install PyOpenGL==3.1.7
python -m pip install -e ./rand_param_envs
```

**For Hopper and Walker environments**, MuJoCo131 is required. Simply install it the same way as MuJoCo200. To switch between different MuJoCo versions:

```bash
cp -r ./pkg/.mujoco ~/
vim ~/.bashrc
source ~/.bashrc

export MUJOCO_PY_MJPRO_PATH=~/.mujoco/mjpro${VERSION_NUM}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro${VERSION_NUM}/bin

use_mujoco() {
    case "$1" in
        131|150) ;;
        *)
            echo "用法: use_mujoco 131 或 use_mujoco 150"
            return 1
            ;;
    esac

    local clean_path=""
    local path_item
    local -a path_items

    IFS=':' read -r -a path_items <<< "${LD_LIBRARY_PATH:-}"

    for path_item in "${path_items[@]}"; do
        case "$path_item" in
            "$HOME/.mujoco/mjpro131/bin"|\
            "$HOME/.mujoco/mjpro150/bin"|\
            "$HOME/.mujoco/mjpro/bin"|"")
                ;;
            *)
                clean_path="${clean_path:+$clean_path:}$path_item"
                ;;
        esac
    done

    export VERSION_NUM="$1"
    export MUJOCO_PY_MJPRO_PATH="$HOME/.mujoco/mjpro${VERSION_NUM}"
    export MUJOCO_PY_MJKEY_PATH="$HOME/.mujoco/mjkey.txt"
    export LD_LIBRARY_PATH="$MUJOCO_PY_MJPRO_PATH/bin${clean_path:+:$clean_path}"

    echo "MuJoCo 已切换到 ${VERSION_NUM}"
    echo "MUJOCO_PY_MJPRO_PATH=$MUJOCO_PY_MJPRO_PATH"
}

```

## Data Generation

Example of training behavior policies on multiple tasks:

```bash
python policy_train.py ./configs/ant-dir.json --gpu 0
```

It will run SAC to train a policy on each task, you can modify `self.work_dir` of `Workspace` in `rlkit/torch/sac/pytorch_sac/train.py` to specify the directory to save the trained policies.

Generate trajectories from trained policies:

```bash
python policy_eval.py --config ./configs/ant-dir.json
```

Data will be saved in `self.work_dir/gentle_data/$env_name/$goal_idx{i}`

## Training GENTLE

The environment configs are in `./configs`. Set `algo_params.data_dir` to the local offline dataset before running. Use the same environment config and seed for dynamics pretraining, context encoder/decoder pretraining, and policy training:

```bash
python pretrain_dynamics.py ./configs/ant-dir.json --gpu 0 --seed_list 0
python pretrain_encoder_decoder.py ./configs/ant-dir.json --gpu 0 --seed_list 0
python train_gentle.py ./configs/ant-dir.json --gpu 0 --seed_list 0
```

Repeat `--seed_list` for additional seeds after preparing their pretrained models. Context checkpoints are loaded from `encoder_decoder/<env_name>/expert_seed<seed>/`; `--path_to_weights` can select another compatible directory containing `context_encoder.pth` and `context_decoder.pth` (the legacy filenames `encoder.pth` and `decoder.pth` are also accepted). Semantic interpolation changes sampling, not the encoder/decoder architecture, so compatible existing context checkpoints can be reused. Changes to latent size, network size, or `use_next_obs_in_context` require compatible checkpoints or new pretraining. Active semantic interpolation fails early if context checkpoints are missing: optimizing paths through a randomly initialized frozen decoder would not give meaningful task geometry.

Logs will be written below `./logs/ant-dir/gentle/`.

## Semantic task interpolation

For matched global / semantic-path0 / semantic-path8 experiments, use the standalone profiles and commands in [configs/interpolation-ablation/README.md](configs/interpolation-ablation/README.md). These profiles fix task counts and RL settings and disable recon, cycle, and semantic quality weighting so virtual RL samples use only the linear decay schedule. They are separate from the historical profiles below.

The six environment configs and their `-vt-decay` / `-vt-decay-cycle-critic` variants use `virtual_task_generation_mode="semantic"`. `SemanticTaskInterpolator` builds nearby task connections using real task prototypes and offline state/action support, then optimizes a discrete latent path using changes in the frozen decoder's normalized reward and dynamics predictions. Its path endpoints, prototypes, and probes stay fixed during each optimization; the cached geometry is refreshed periodically as the encoder learns. Held-out probes check a path before it supplies virtual tasks.

Support is a nearest-neighbor heuristic in standardized observation/action coordinates, not a guarantee that an interpolated task exists in the environment. The decoder is a single model; `algo_params.ensemble_size` controls the separate pretrained task dynamics and does not provide uncertainty estimates for arbitrary virtual task embeddings. The supplied settings are conservative starting profiles, not performance-tuned results or guarantees that paths lie on the true task manifold. Measure held-out task return, valid-edge coverage, and generated-data quality before increasing virtual-task influence.

| Environment | Virtual tasks per sampling call | Neighbors | Fit probes per task | Support radius | Trust radius | Path sampling interval | Initial virtual RL weight |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Point-Robot | 4 | 3 | 64 | 1.0 | 0.10 | 0.10–0.90 | 0.5 |
| Ant-Dir | 4 | 2 | 64 | 1.0 | 0.10 | 0.10–0.90 | 0.5 |
| Cheetah-Vel | 4 | 2 | 64 | 1.0 | 0.10 | 0.10–0.90 | 0.5 |
| Hopper-Rand-Params | 4 | 2 | 96 | 0.75 | 0.05 | 0.05–0.25 | 0.2 |
| Walker-Rand-Params | 4 | 2 | 96 | 0.75 | 0.05 | 0.05–0.25 | 0.2 |
| Cheetah-Dir | 0 | 2 | 64 | 1.0 | 0.10 | disabled | 0.0 |

Cheetah-Dir defines only forward/backward tasks. Its three configs and the bare `default.py` therefore disable virtual task generation (`n_vt=0`) and virtual transition loss (`virtual_transition_loss_weight=0`), rather than inventing a continuous family between the two directions. Real-task training and consistency still run. For continuous-task environments, the JSON profiles explicitly enable four virtual tasks. Dynamics environments use narrower support, a smaller trust region, and samples closer to a randomly chosen end of the graph path. The sampling interval is a fraction of the complete path's semantic arc length, not a guaranteed Euclidean distance from an endpoint; a path can contain multiple supported edges.

All profiles disable actor behavior cloning on virtual transitions: an offline action need not be an expert action for the new task. The original virtual-weight schedules and the `-vt-decay-cycle-critic` variants' actor-Q disablement remain in place. Those variants retain optional cycle-based weighting; low cycle error measures encoder/decoder self-consistency and is not independent evidence of physical task validity.

`virtual_transition_use_semantic_weight` defaults to true for backward compatibility. It controls the semantic sampler's endpoint-quality discount on virtual RL samples independently of `virtual_transition_use_recon_weight`. Turning it off preserves graph/support checks and only removes that sample-weight factor; the CLI override is `--virtual_transition_use_semantic_weight false`.

The following settings live under `algo_params`. The four semantic parameters shown in the override example below can also be changed on the training command line. `--n_vt` overrides the number of virtual tasks per sampling call (a nonnegative integer); zero disables their generation. `--virtual_semantic_neighbors` limits graph-neighbor candidates per real task, not the number of tasks simultaneously averaged into one embedding. `--M` applies only to the legacy local/global samplers.

| Parameter | Reward-task default | Meaning |
| --- | ---: | --- |
| `virtual_semantic_neighbors` | 2 (Point: 3) | Maximum nearby candidate task connections per anchor. |
| `virtual_semantic_probe_batch_size` | 64 | Per-task fit-probe count; a second batch of this size is reserved for validation. |
| `virtual_semantic_refresh_interval` | 100 | Refresh cached prototypes, support probes, and paths every this many gradient steps, not outer training iterations. |
| `virtual_semantic_support_radius` | 1.0 | Nearest-neighbor RMS distance threshold in standardized observation/action coordinates. |
| `virtual_semantic_min_shared_probes` | 8 | Minimum number of probes supported by both endpoint tasks. |
| `virtual_semantic_max_latent_distance_ratio` | 2.0 | Limit endpoint separation relative to nearby task spacing. |
| `virtual_semantic_max_reconstruction_error` | 1.0 | Maximum normalized endpoint reconstruction error for an accepted connection. |
| `virtual_semantic_reward_weight` | 1.0 | Weight of normalized reward differences in path energy. |
| `virtual_semantic_dynamics_weight` | 0.0 | Weight of normalized state-change predictions (decoder next state minus current state); Hopper/Walker set this to 1.0. |
| `virtual_semantic_path_nodes` | 7 | Path nodes including the two fixed endpoints. |
| `virtual_semantic_path_steps` | 8 | Optimization steps for each cached path. |
| `virtual_semantic_path_lr` | 0.02 | Path optimizer learning rate; model parameters are not updated by it. |
| `virtual_semantic_trust_radius` | 0.1 | Maximum deviation from the initial path, relative to endpoint separation. |
| `virtual_semantic_latent_weight` | 0.05 | Latent-distance regularizer against decoder-insensitive shortcuts. |
| `virtual_semantic_validation_ratio` | 1.25 | Allowed held-out path energy relative to the initial straight path. |
| `virtual_semantic_alpha_min` | 0.1 | Lower endpoint of the accepted path's sampling interval. |
| `virtual_semantic_alpha_max` | 0.9 | Upper endpoint of the accepted path's sampling interval. |

For example, a single-seed run with explicit semantic overrides is:

```bash
python train_gentle.py ./configs/point-robot.json --gpu 0 --seed_list 0 --n_vt 4 --virtual_task_generation_mode semantic --virtual_semantic_path_steps 8 --virtual_semantic_refresh_interval 100 --virtual_semantic_neighbors 3 --virtual_semantic_support_radius 1.0
```

If no trustworthy connection passes the support and endpoint-reconstruction checks, semantic sampling skips virtual tasks. An optimized path that fails held-out validation may fall back to its supported straight reference path; this is recorded separately and should not be counted as successful nonlinear refinement. It does not enable interpolation between rejected or unsupported task pairs.

Inspect these training statistics alongside real-task reconstruction error, virtual-task counts, and held-out task return:

| Log key | Interpretation |
| --- | --- |
| `virtual_semantic_edges` / `virtual_semantic_paths` | Number of accepted task connections and available paths. |
| `virtual_semantic_rejected_support` | Connections rejected for insufficient shared offline support. |
| `virtual_semantic_rejected_reconstruction` | Connections rejected by the endpoint decoder reconstruction check. |
| `virtual_semantic_rejected_latent` | Connections rejected by the latent-distance constraint. |
| `virtual_semantic_refinement_accepted` / `virtual_semantic_refinement_rejected` | Successful versus rejected path refinements. |
| `virtual_semantic_reference_fallbacks` | Supported straight reference paths retained instead of accepted optimized paths. |
| `virtual_semantic_sampled` / `virtual_semantic_rejected_no_path` | Generated virtual embeddings versus skips because no accepted path is available. |
| `virtual_semantic_refresh_count` | Geometry refresh count. |

A zero virtual-task count can be an intended skip, not a crash. Do not automatically loosen support thresholds just to obtain nonzero counts; first inspect data overlap and decoder reconstruction. `virtual_semantic_candidate_edges`, `virtual_semantic_supported_edges`, and `virtual_semantic_refinement_attempts` provide additional context for acceptance rates.

The previous modes remain available for ablations on continuous-task profiles:

```bash
python train_gentle.py ./configs/point-robot.json --gpu 0 --seed_list 0 --virtual_task_generation_mode local
python train_gentle.py ./configs/point-robot.json --gpu 0 --seed_list 0 --virtual_task_generation_mode global --M 2
python train_gentle.py ./configs/point-robot.json --gpu 0 --seed_list 0 --virtual_task_generation_mode gaussian
```

`M`, `beta`, `virtual_interpolation_lambda_max`, and `virtual_gaussian_noise_std` keep their previous meanings in the corresponding legacy modes; they do not tune semantic paths. The global mode can extrapolate when `beta > 1`. The commands above share the new profiles' RL weights and schedules, so they compare samplers rather than reproduce the repository's previous experiment settings.

The embedding exporter reads the saved experiment's generation mode. Semantic export loads both `context_encoder_itr_<epoch>.pth` and `context_decoder_itr_<epoch>.pth` and uses the same path sampler as training. It reports an error when no supported paths exist instead of silently exporting linear mixtures. Configurations with `n_vt=0` have no virtual embeddings to export.

Run the synthetic sampler and training-integration checks in a PyTorch environment with pytest:

```bash
python -m pytest tests -q
```

These checks cover constrained nonlinear paths, support rejection, paired transition inputs, decoder gradients, cached geometry, and real actor/critic/encoder updates without requiring a MuJoCo environment. They do not measure full benchmark training performance.

visualize
```
python replot_tsne.py ^
  --real "logs/<env_name>/<algo_name>/seed0/<timestamp>/saved_zs/offline_z_train_itr_499.npy" ^
  --virtual "logs/<env_name>/<algo_name>/seed0/<timestamp>/saved_zs/offline_virtual_z_train_itr_499.npy" ^
  --output "logs/<env_name>/<algo_name>/seed0/<timestamp>/figures/offline_z_train_itr_499_replot.png" ^
  --title "offline_z_train_itr_499" ^
  --tsne-seed 0

python replot_tsne.py --real ./logs/point-robot/gentle/seed0/2026_04_02_20_10_56/saved_zs/offline_z_train_itr_499.npy --virtual ./logs/point-robot/gentle/seed0/2026_04_02_20_10_56/saved_zs/offline_virtual_z_train_itr_499.npy --output ./logs/point-robot/gentle/seed0/2026_04_02_20_10_56/figures/offline_z_train_itr_499_replot.png --title "offline_z_train_itr_499" --tsne-seed 0 


python export_virtual_embeddings.py \
  --log-dir /root/tievnas/YinCH/GENTLE-consistant/logs/point-robot/gentle/seed0/2026_04_02_11_17_00 \
  --epoch 499 \
  --gpu 0

python plot_tb_seed_average.py --root logs/point-robot/gentle --experiment "2026_04_20_12_55_45_batch512"  --all-tags --show-seeds --output-dir figures/point-tobot_batch512_all_tags

```

## Reference

```bash
@inproceedings{gentle,
  author={Renzhe Zhou, Chen-Xiao Gao, Zongzhang Zhang, Yang Yu},
  title={Generalizable Task Representation Learning for Offline Meta-Reinforcement Learning with Data Limitations},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2024}
}
```

