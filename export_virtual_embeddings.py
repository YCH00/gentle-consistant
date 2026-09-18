import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Dirichlet

from configs.default import default_config
from pretrain_encoder_decoder import (
    build_context,
    build_train_buffer,
    deep_update_dict,
    global_seed,
    infer_task_embedding,
    sample_context_batch,
)
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.autoencoder import MlpEncoder, MlpDecoder
from rlkit.torch.task_interpolation import SemanticTaskInterpolator
import rlkit.torch.pytorch_util as ptu


def infer_seed_from_log_dir(log_dir: Path):
    for part in log_dir.parts:
        match = re.fullmatch(r"seed(\d+)", part)
        if match is not None:
            return int(match.group(1))
    return None


def resolve_epoch(log_dir: Path, epoch_arg: str) -> int:
    if epoch_arg != "latest":
        return int(epoch_arg)

    checkpoint_paths = sorted(log_dir.glob("context_encoder_itr_*.pth"))
    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(f"No context encoder checkpoints found under {log_dir}.")

    epochs = []
    for checkpoint_path in checkpoint_paths:
        match = re.fullmatch(r"context_encoder_itr_(\d+)\.pth", checkpoint_path.name)
        if match is not None:
            epochs.append(int(match.group(1)))
    if len(epochs) == 0:
        raise FileNotFoundError(f"Could not parse checkpoint epochs under {log_dir}.")
    return max(epochs)


def load_variant(log_dir: Path):
    variant_path = log_dir / "variant.json"
    if not variant_path.exists():
        raise FileNotFoundError(f"Missing variant.json under {log_dir}.")

    with open(variant_path, "r", encoding="utf-8") as f:
        saved_variant = json.load(f)

    variant = deep_update_dict(saved_variant, copy.deepcopy(default_config))
    # Historical checkpoints without a mode used this exporter's Dirichlet sampler.
    if 'virtual_task_generation_mode' not in saved_variant.get('algo_params', {}):
        variant['algo_params']['virtual_task_generation_mode'] = 'global'
    return variant


def build_context_encoder(variant, obs_dim: int, action_dim: int) -> MlpEncoder:
    latent_dim = variant["latent_size"]
    net_size = variant["net_size"]
    use_next_obs_in_context = variant["algo_params"]["use_next_obs_in_context"]
    use_information_bottleneck = variant["algo_params"]["use_information_bottleneck"]

    input_dim = 2 * obs_dim + action_dim + 1 if use_next_obs_in_context else obs_dim + action_dim + 1
    output_dim = latent_dim * 2 if use_information_bottleneck else latent_dim

    return MlpEncoder(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=input_dim,
        output_size=output_dim,
        output_activation=torch.tanh,
        batch_attention=False,
    ).to(ptu.device)


@torch.no_grad()
def sample_virtual_embeddings(
    train_buffer,
    train_tasks,
    context_encoder,
    latent_dim: int,
    use_next_obs_in_context: bool,
    use_information_bottleneck: bool,
    n_vt: int,
    mixing_tasks: int,
    beta: float,
    batch_size: int,
    n_points: int,
    generation_mode: str = 'global',
    context_decoder=None,
    algo_params=None,
):
    if n_vt <= 0:
        raise ValueError("n_vt must be > 0 to export virtual task embeddings.")
    if n_points <= 0 or batch_size <= 0:
        raise ValueError('n_points and batch_size must be positive')
    algo_params = {} if algo_params is None else algo_params
    if generation_mode not in ('semantic', 'global', 'local', 'gaussian'):
        raise ValueError('Unknown virtual task generation mode: {}'.format(generation_mode))

    def encode_tasks(task_indices):
        obs, actions, rewards, next_obs, _ = sample_context_batch(
            train_buffer, task_indices, batch_size
        )
        return infer_task_embedding(
            context_encoder=context_encoder,
            context=build_context(obs, actions, rewards, next_obs, use_next_obs_in_context),
            latent_dim=latent_dim,
            use_information_bottleneck=use_information_bottleneck,
        )

    if generation_mode == 'semantic':
        if context_decoder is None:
            raise ValueError('semantic export requires the matching context decoder checkpoint')
        interpolator = SemanticTaskInterpolator(
            context_decoder, use_next_obs_in_context,
            **{key: value for key, value in algo_params.items() if key.startswith('virtual_semantic_')},
        )
        probe_count = 2 * int(algo_params.get('virtual_semantic_probe_batch_size', 64))
        transitions = sample_context_batch(train_buffer, train_tasks, probe_count)
        batches = dict(zip(
            ('observations', 'actions', 'rewards', 'next_observations', 'terminals'), transitions
        ))
        interpolator.refresh(encode_tasks(train_tasks), batches)
        virtual_zs = []
        for _ in range(n_points):
            samples = interpolator.sample(n_vt, batch_size)
            if samples is None:
                raise RuntimeError(
                    'No supported semantic task paths found; no embeddings exported. '
                    'Check model reconstruction and shared-data coverage. Statistics: {}'.format(
                        interpolator.stats
                    )
                )
            virtual_zs.append(samples['embeddings'].cpu().numpy())
        print('Semantic interpolation statistics:', interpolator.stats)
        return np.concatenate(virtual_zs, axis=0)[np.newaxis, ...]

    virtual_zs = []
    for _ in range(n_points):
        if generation_mode == 'gaussian':
            indices = np.random.choice(train_tasks, n_vt, replace=True)
            real_z = encode_tasks(indices)
            noise_std = float(algo_params.get('virtual_gaussian_noise_std', 0.05))
            virtual_zs.append((real_z + torch.randn_like(real_z) * noise_std).cpu().numpy())
            continue
        if generation_mode == 'local':
            if len(train_tasks) < 2:
                raise ValueError('local interpolation requires at least two training tasks')
            real_z = encode_tasks(train_tasks)
            distances = torch.cdist(real_z, real_z)
            distances.fill_diagonal_(float('inf'))
            k = min(max(1, mixing_tasks), len(train_tasks) - 1)
            neighbors = distances.topk(k, largest=False, dim=1).indices
            candidates = []
            for _ in range(n_vt * 4):
                anchor = np.random.randint(len(train_tasks))
                weights = torch.rand(1, k, device=real_z.device)
                weights = weights / weights.sum().clamp(min=1e-8)
                neighbor = weights @ real_z[neighbors[anchor]]
                ratio = torch.rand(1, 1, device=real_z.device) * max(
                    0.0, float(algo_params.get('virtual_interpolation_lambda_max', 0.2))
                )
                candidate = (1 - ratio) * real_z[anchor:anchor + 1] + ratio * neighbor
                max_distance = algo_params.get('virtual_interpolation_max_distance')
                if max_distance is not None and (real_z - candidate).norm(dim=-1).min() > max_distance:
                    continue
                candidates.append(candidate)
                if len(candidates) == n_vt:
                    break
            if not candidates:
                raise RuntimeError('No local virtual tasks passed the distance filter')
            virtual_zs.append(torch.cat(candidates).cpu().numpy())
            continue
        mixed_embeddings = []
        for _ in range(n_vt):
            mixing_tasks = max(1, int(mixing_tasks))
            task_indices = np.random.choice(train_tasks, mixing_tasks, replace=True)
            obs, actions, rewards, next_obs, _ = sample_context_batch(train_buffer, task_indices, batch_size)
            context = build_context(obs, actions, rewards, next_obs, use_next_obs_in_context)
            task_embeddings = infer_task_embedding(
                context_encoder=context_encoder,
                context=context,
                latent_dim=latent_dim,
                use_information_bottleneck=use_information_bottleneck,
            )
            alpha = Dirichlet(torch.ones(mixing_tasks, device=ptu.device)).sample().unsqueeze(0)
            alpha = alpha * beta - (beta - 1) / mixing_tasks
            mixed_embeddings.append(alpha @ task_embeddings)
        virtual_zs.append(torch.cat(mixed_embeddings, dim=0).cpu().numpy())

    virtual_zs = np.concatenate(virtual_zs, axis=0)
    return virtual_zs[np.newaxis, ...]


def main():
    parser = argparse.ArgumentParser(description="Export offline virtual task embeddings from a saved checkpoint.")
    parser.add_argument("--log-dir", required=True, help="Experiment log directory, e.g. logs/point-robot/gentle/seed0/<timestamp>")
    parser.add_argument("--epoch", default="latest", help="Checkpoint epoch to load. Use an integer or 'latest'.")
    parser.add_argument("--seed", type=int, default=None, help="Override the seed parsed from the log directory.")
    parser.add_argument("--gpu", type=int, default=None, help="GPU id to use. By default export runs on CPU.")
    parser.add_argument("--n-points", type=int, default=400, help="Number of sampling rounds for visualization export.")
    parser.add_argument("--output", default=None, help="Output .npy path. Defaults to saved_zs/offline_virtual_z_train_itr_<epoch>.npy")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    epoch = resolve_epoch(log_dir, args.epoch)
    seed = args.seed if args.seed is not None else infer_seed_from_log_dir(log_dir)
    if seed is None:
        raise ValueError("Could not infer seed from log-dir. Please pass --seed explicitly.")

    variant = load_variant(log_dir)
    use_gpu = args.gpu is not None and torch.cuda.is_available()
    gpu_id = args.gpu if args.gpu is not None else 0
    ptu.set_gpu_mode(use_gpu, gpu_id)

    env = NormalizedBoxEnv(ENVS[variant["env_name"]](**variant["env_params"]))
    global_seed(seed)
    env.seed(seed)

    tasks = env.get_all_task_idx()
    train_tasks = list(tasks[:variant["n_train_tasks"]])
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    latent_dim = variant["latent_size"]

    train_buffer, _ = build_train_buffer(variant, env, train_tasks, obs_dim)
    context_encoder = build_context_encoder(variant, obs_dim, action_dim)

    checkpoint_path = log_dir / f"context_encoder_itr_{epoch}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=ptu.device)
    context_encoder.load_state_dict(state_dict)
    context_encoder.eval()

    generation_mode = variant['algo_params']['virtual_task_generation_mode']
    context_decoder = None
    if generation_mode == 'semantic':
        decoder_path = log_dir / f'context_decoder_itr_{epoch}.pth'
        if not decoder_path.exists():
            raise FileNotFoundError(
                f'Semantic export needs the matching decoder checkpoint: {decoder_path}'
            )
        context_decoder = MlpDecoder(
            hidden_size=variant['net_size'], num_hidden_layers=3,
            z_dim=latent_dim, action_dim=action_dim, obs_dim=obs_dim, reward_dim=1,
            use_next_obs_in_context=variant['algo_params']['use_next_obs_in_context'],
        ).to(ptu.device)
        context_decoder.load_state_dict(torch.load(decoder_path, map_location=ptu.device))
        context_decoder.eval()
        context_decoder.requires_grad_(False)

    offline_virtual_zs = sample_virtual_embeddings(
        train_buffer=train_buffer,
        train_tasks=train_tasks,
        context_encoder=context_encoder,
        latent_dim=latent_dim,
        use_next_obs_in_context=variant["algo_params"]["use_next_obs_in_context"],
        use_information_bottleneck=variant["algo_params"]["use_information_bottleneck"],
        n_vt=variant["algo_params"]["n_vt"],
        mixing_tasks=variant["algo_params"]["M"],
        beta=variant["algo_params"]["beta"],
        batch_size=max(
            variant['algo_params']['online_sample_num'],
            variant['algo_params']['embedding_batch_size'],
        ) if generation_mode == 'semantic' else variant['algo_params']['online_sample_num'],
        n_points=args.n_points,
        generation_mode=generation_mode,
        context_decoder=context_decoder,
        algo_params=variant['algo_params'],
    )

    output_path = Path(args.output) if args.output is not None else log_dir / "saved_zs" / f"offline_virtual_z_train_itr_{epoch}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, offline_virtual_zs)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Saved virtual task embeddings to: {output_path}")
    print(f"Embedding shape: {offline_virtual_zs.shape}")


if __name__ == "__main__":
    main()
