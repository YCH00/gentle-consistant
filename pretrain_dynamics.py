import os
import numpy as np
import click
import json, time
import torch
import random
import multiprocessing as mp
from itertools import product
import glob, ast

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.multi_task_dynamics import MultiTaskDynamics

import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
from numpy.random import default_rng
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from pathlib import Path
from rlkit.data_management.offline_sources import trajectory_paths, load_trajectory, dynamics_path, bind_tasks, sha256

rng = default_rng()

def global_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def experiment(variant, seed=None):
    raw_env = ENVS[variant['env_name']](**variant['env_params'])
    bind_tasks(raw_env, variant)
    env = NormalizedBoxEnv(raw_env)
    
    if seed is not None:
        global_seed(seed)
        env.seed(seed)

    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    obs_normalizer = ptu.RunningMeanStd(shape=obs_dim)

    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    net_size = variant['net_size']
    use_next_obs_in_context = variant['algo_params']['use_next_obs_in_context']
    
    if use_next_obs_in_context:
        task_dynamics =  MultiTaskDynamics(num_tasks=variant['n_train_tasks'], 
                                     hidden_size=net_size, 
                                     num_hidden_layers=3, 
                                     action_dim=action_dim, 
                                     obs_dim=obs_dim,
                                     reward_dim=1,
                                     use_next_obs_in_context=use_next_obs_in_context,
                                     ensemble_size=variant['algo_params']['ensemble_size'],
                                     dynamics_weight_decay=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5])
    else:
        task_dynamics = MultiTaskDynamics(num_tasks=variant['n_train_tasks'], 
                                     hidden_size=net_size, 
                                     num_hidden_layers=2, 
                                     action_dim=action_dim, 
                                     obs_dim=obs_dim,
                                     reward_dim=1,
                                     use_next_obs_in_context=use_next_obs_in_context,
                                     ensemble_size=variant['algo_params']['ensemble_size'],
                                     dynamics_weight_decay=[2.5e-5, 5e-5, 7.5e-5])
    task_dynamics.to(device=ptu.device)
    train_tasks = list(tasks[:variant['n_train_tasks']])
    train_buffer = MultiTaskReplayBuffer(variant['algo_params']['replay_buffer_size'], env, train_tasks, 1)

    train_paths, train_task_idxs = trajectory_paths(variant['algo_params'], train_tasks)

    obs_train_lst = []
    action_train_lst = []
    reward_train_lst = []
    next_obs_train_lst = []
    terminal_train_lst = []
    task_train_lst = []

    for train_path, train_task_idx in zip(train_paths, train_task_idxs):
        trj_npy = load_trajectory(train_path, variant['algo_params'])
        obs_train_lst += list(trj_npy[:, 0])
        action_train_lst += list(trj_npy[:, 1])
        reward_train_lst += list(trj_npy[:, 2])
        next_obs_train_lst += list(trj_npy[:, 3])
        terminal = [0 for _ in range(trj_npy.shape[0])]
        terminal[-1] = 1
        terminal_train_lst += terminal
        task_train = [train_task_idx for _ in range(trj_npy.shape[0])]
        task_train_lst += task_train
        
    obs_normalizer.update(obs_train_lst)
    env.update_obs_mean_var(obs_normalizer.mean, obs_normalizer.var)
    obs_train_lst = obs_normalizer.forward(obs_train_lst)
    next_obs_train_lst = obs_normalizer.forward(next_obs_train_lst)
        
        # load training buffer
    for i, (
            task_train, obs, action, reward, next_obs, terminal,
    ) in enumerate(zip(
            task_train_lst,
            obs_train_lst,
            action_train_lst,
            reward_train_lst,
            next_obs_train_lst,
            terminal_train_lst,
    )):
        train_buffer.add_sample(task_train, obs, action, reward, terminal, next_obs, **{'env_info': {}},)

    for task_idx in train_tasks:
        data = train_buffer.get_all_data(task_idx)
        task_dynamics.set_task_idx(task_idx)
        task_dynamics.train(data, max_epochs=variant.get('dynamics_max_epochs'))
        print(f"Task {task_idx} finished training")
    save_dir_path = dynamics_path(variant, seed)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    task_dynamics.save(str(save_dir_path))
    if variant['algo_params'].get('mixed_artifacts_dir'):
        receipt = {'dataset_sha256': sha256(Path(variant['algo_params']['mixed_artifacts_dir']) / 'dataset_manifest.json')}
        (save_dir_path / 'data_receipt.json').write_text(json.dumps(receipt), encoding='utf-8')


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--seed_list', multiple=True, type=int, default=[0,1,2,3])

def main(config, gpu, seed_list):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    if len(seed_list) > 1:
        if isinstance(seed_list, str):
            seed_list = ast.literal_eval(seed_list)
        p = mp.Pool(len(seed_list))
        p.starmap(experiment, product([variant], seed_list))
    else:
        experiment(variant, seed=seed_list[0])

if __name__ == "__main__":
    main()



