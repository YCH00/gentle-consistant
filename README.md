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

The configration files to run GENTLE is in `./configs`. For example, to train GENTLE on Ant-Dir, first you need to pretrain the dynamics model:
```bash
python pretrain_dynamics.py ./configs/ant-dir.json 
```
Then run:
```bash
python train_gentle.py ./configs/ant-dir.json
```

Logs will be written to `./logs/ant-dir/gentle/`

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

