import os
import numpy as np
import click
import json
import torch
import random
import multiprocessing as mp

from pathlib import Path
from itertools import product
from torch.utils.tensorboard import SummaryWriter

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.autoencoder import MlpEncoder, MlpDecoder
from rlkit.torch.multi_task_dynamics import MultiTaskDynamics
from rlkit.torch.sac.agent import Agent
from rlkit.torch.sac.policies import ContextPolicyWrapper
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
from numpy.random import default_rng
from rlkit.torch.algo.gentle import GENTLE

rng = default_rng()

def global_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _resolve_weights_dir(variant, seed):
    weights_dir = variant.get('path_to_weights')
    if weights_dir is not None:
        weights_dir = Path(weights_dir).expanduser()
        if not weights_dir.is_absolute():
            weights_dir = Path(__file__).parent.absolute() / weights_dir
        return weights_dir
    return Path(__file__).parent.absolute() / 'encoder_decoder' / variant['env_name'] / f'expert_seed{seed}'


def _find_weight_file(weights_dir, candidate_filenames):
    for filename in candidate_filenames:
        weight_path = weights_dir / filename
        if weight_path.exists():
            return weight_path
    return None


def _load_pretrained_context_models(context_encoder, context_decoder, variant, seed):
    weights_dir = _resolve_weights_dir(variant, seed)
    encoder_path = _find_weight_file(weights_dir, ['context_encoder.pth', 'encoder.pth'])
    decoder_path = _find_weight_file(weights_dir, ['context_decoder.pth', 'decoder.pth'])

    if encoder_path is None and decoder_path is None and variant.get('path_to_weights') is None:
        print(f'No pretrained context weights found at {weights_dir}, continue with random initialization.')
        return

    if encoder_path is None or decoder_path is None:
        raise FileNotFoundError(
            f'Incomplete pretrained context weights in {weights_dir}. '
            f'Found encoder={encoder_path is not None}, decoder={decoder_path is not None}.'
        )

    try:
        context_encoder.load(encoder_path)
        context_decoder.load(decoder_path)
    except RuntimeError as error:
        raise RuntimeError(
            'Failed to load pretrained context encoder/decoder. '
            'If use_next_obs_in_context was changed, rerun pretrain_encoder_decoder.py '
            'with the same config before training GENTLE.'
        ) from error
    print(f'Loaded pretrained context encoder from {encoder_path}')
    print(f'Loaded pretrained context decoder from {decoder_path}')

def experiment(variant, seed=None):
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    
    if seed is not None:
        global_seed(seed)
        env.seed(seed)

    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    obs_normalizer = ptu.RunningMeanStd(shape=obs_dim)

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    use_next_obs_in_context = variant['algo_params']['use_next_obs_in_context']
    variant['algo_params']['context_dim'] = context_encoder_input_dim

    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])

    context_encoder = MlpEncoder(
            hidden_sizes=[net_size, net_size, net_size],
            input_size=context_encoder_input_dim,
            output_size=context_encoder_output_dim,
            output_activation=torch.tanh,
            batch_attention=False,
    )

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )

    context_decoder = MlpDecoder(hidden_size=net_size,
                                    num_hidden_layers=3,
                                    z_dim=latent_dim,
                                    action_dim=action_dim,
                                    obs_dim=obs_dim,
                                    reward_dim=1,
                                    use_next_obs_in_context=use_next_obs_in_context)
    _load_pretrained_context_models(context_encoder, context_decoder, variant, seed)

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
    dynamics_dir_path = Path(__file__).parent.absolute()/'dynamics'/variant['env_name']/f'expert_seed{seed}'
    try:
        task_dynamics.load(str(dynamics_dir_path))
    except RuntimeError as error:
        raise RuntimeError(
            'Failed to load pretrained task dynamics. '
            'If use_next_obs_in_context was changed, rerun pretrain_dynamics.py '
            'with the same config before training GENTLE.'
        ) from error

    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )

    context_policy = ContextPolicyWrapper(policy, latent_dim)

    agent = Agent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    agent = [agent, context_policy]

    algorithm = GENTLE(
                    env=env,
                    train_tasks=list(tasks[:variant['n_train_tasks']]),
                    eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
                    nets=[agent, qf1, qf2, context_decoder, task_dynamics],
                    latent_dim=latent_dim,
                    obs_normalizer=obs_normalizer,
                    **variant['algo_params']
    )
    # optional GPU mode
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = variant.get('exp_name')
    if exp_id is None and DEBUG:
        exp_id = 'debug'
    experiment_log_dir = setup_logger(
        variant['env_name'],
        variant=variant,
        exp_id=exp_id,
        base_log_dir=variant['util_params']['base_log_dir'],
        seed=seed,
        snapshot_mode="all",
        algo_name=variant['output_prefix']+variant['algo_type']
    )

    tb_writer = SummaryWriter(experiment_log_dir)
    algorithm.train(tb_writer)

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
@click.option('--debug', default=0)
@click.option('--algo_type', default='gentle')  
@click.option('--seed_list', multiple=True, type=int, default=[0,1,2,3])
@click.option('--output_prefix', default='')
@click.option('--exp_name', default=None)
@click.option('--path_to_weights', default=None)
@click.option('--M', 'virtual_neighbor_candidates', type=int, default=None)
@click.option('--virtual_interpolation_lambda_max', type=float, default=None)
@click.option('--consistency_use_policy_relabel_data', type=bool, default=None)
@click.option('--virtual_transition_use_policy_actions', type=bool, default=None)
@click.option('--virtual_transition_loss_weight', type=float, default=None)
@click.option('--virtual_transition_weight_schedule', type=click.Choice(['constant', 'linear_decay']), default=None)
@click.option('--virtual_transition_weight_decay_start_itr', type=int, default=None)
@click.option('--virtual_transition_weight_decay_end_itr', type=int, default=None)
@click.option('--virtual_transition_final_loss_weight', type=float, default=None)
@click.option('--virtual_transition_train_policy', type=bool, default=None)
@click.option('--virtual_transition_train_policy_q', type=bool, default=None)
@click.option('--virtual_transition_train_policy_bc', type=bool, default=None)
@click.option('--virtual_task_generation_mode', type=click.Choice(['local', 'global', 'gaussian']), default=None)
def main(
    config,
    gpu,
    debug,
    algo_type,
    seed_list,
    output_prefix,
    exp_name,
    path_to_weights,
    virtual_neighbor_candidates,
    virtual_interpolation_lambda_max,
    consistency_use_policy_relabel_data,
    virtual_transition_use_policy_actions,
    virtual_transition_loss_weight,
    virtual_transition_weight_schedule,
    virtual_transition_weight_decay_start_itr,
    virtual_transition_weight_decay_end_itr,
    virtual_transition_final_loss_weight,
    virtual_transition_train_policy,
    virtual_transition_train_policy_q,
    virtual_transition_train_policy_bc,
    virtual_task_generation_mode,
):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    variant['util_params']['debug'] = debug
    variant['algo_type'] = algo_type
    variant['output_prefix'] = output_prefix
    variant['exp_name'] = exp_name
    variant['util_params']['base_log_dir'] = './logs'
    if path_to_weights is not None:
        variant['path_to_weights'] = path_to_weights
    if virtual_neighbor_candidates is not None:
        variant['algo_params']['M'] = virtual_neighbor_candidates
    if virtual_interpolation_lambda_max is not None:
        variant['algo_params']['virtual_interpolation_lambda_max'] = virtual_interpolation_lambda_max
    if consistency_use_policy_relabel_data is not None:
        variant['algo_params']['consistency_use_policy_relabel_data'] = consistency_use_policy_relabel_data
    if virtual_transition_use_policy_actions is not None:
        variant['algo_params']['virtual_transition_use_policy_actions'] = virtual_transition_use_policy_actions
    if virtual_transition_loss_weight is not None:
        variant['algo_params']['virtual_transition_loss_weight'] = virtual_transition_loss_weight
    if virtual_transition_weight_schedule is not None:
        variant['algo_params']['virtual_transition_weight_schedule'] = virtual_transition_weight_schedule
    if virtual_transition_weight_decay_start_itr is not None:
        variant['algo_params']['virtual_transition_weight_decay_start_itr'] = virtual_transition_weight_decay_start_itr
    if virtual_transition_weight_decay_end_itr is not None:
        variant['algo_params']['virtual_transition_weight_decay_end_itr'] = virtual_transition_weight_decay_end_itr
    if virtual_transition_final_loss_weight is not None:
        variant['algo_params']['virtual_transition_final_loss_weight'] = virtual_transition_final_loss_weight
    if virtual_transition_train_policy is not None:
        variant['algo_params']['virtual_transition_train_policy'] = virtual_transition_train_policy
        variant['algo_params']['virtual_transition_train_policy_q'] = virtual_transition_train_policy
        variant['algo_params']['virtual_transition_train_policy_bc'] = virtual_transition_train_policy
    if virtual_transition_train_policy_q is not None:
        variant['algo_params']['virtual_transition_train_policy_q'] = virtual_transition_train_policy_q
    if virtual_transition_train_policy_bc is not None:
        variant['algo_params']['virtual_transition_train_policy_bc'] = virtual_transition_train_policy_bc
    if virtual_task_generation_mode is not None:
        variant['algo_params']['virtual_task_generation_mode'] = virtual_task_generation_mode

    # multi-processing
    if len(seed_list) > 1:
        with mp.Pool(processes=len(seed_list)) as pool:
            pool.starmap(experiment, product([variant], seed_list))
            pool.close()
            pool.join()  # 等待所有进程完成
    else:
        experiment(variant, seed=seed_list[0])

if __name__ == "__main__":
    main()



