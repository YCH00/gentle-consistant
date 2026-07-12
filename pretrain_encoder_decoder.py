import json
import os
import random
import glob
import multiprocessing as mp
from itertools import product
from pathlib import Path
from typing import Sequence, Tuple

import click
import numpy as np
import torch
import torch.nn.functional as F

from configs.default import default_config
from rlkit.data_management.env_replay_buffer import MultiTaskContextBuffer, MultiTaskReplayBuffer
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.autoencoder import MlpDecoder, MlpEncoder
from rlkit.torch.multi_task_dynamics import MultiTaskDynamics
import rlkit.torch.pytorch_util as ptu


def global_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def deep_update_dict(fr, to):
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


def extract_task_idx(path: str) -> int:
    task_dir_name = Path(path).parent.name
    return int(task_dir_name.replace("goal_idx", ""))


def load_train_paths(data_dir: str, train_epoch, n_trj: int, train_tasks: Sequence[int]) -> Sequence[str]:
    train_trj_paths = []
    for trj_idx in range(n_trj):
        if train_epoch is None:
            pattern = f"trj_evalsample{trj_idx}_step*.npy"
        else:
            pattern = f"trj_evalsample{trj_idx}_step{int(train_epoch)}.npy"
        train_trj_paths.extend(glob.glob(os.path.join(data_dir, "goal_idx*", pattern)))

    train_paths = [path for path in train_trj_paths if extract_task_idx(path) in train_tasks]
    if len(train_paths) == 0:
        raise FileNotFoundError(
            f"No training trajectories found under {data_dir} for tasks {list(train_tasks)} "
            f"with train_epoch={train_epoch} and n_trj={n_trj}."
        )
    return train_paths


def build_train_buffer(
    variant,
    env,
    train_tasks: Sequence[int],
    obs_dim: int,
) -> Tuple[MultiTaskReplayBuffer, ptu.RunningMeanStd]:
    train_buffer = MultiTaskReplayBuffer(
        variant["algo_params"]["replay_buffer_size"],
        env,
        train_tasks,
        1,
    )
    obs_normalizer = ptu.RunningMeanStd(shape=obs_dim)

    train_paths = load_train_paths(
        data_dir=variant["algo_params"]["data_dir"],
        train_epoch=variant["algo_params"]["train_epoch"],
        n_trj=variant["algo_params"]["n_trj"],
        train_tasks=train_tasks,
    )

    obs_train_lst = []
    action_train_lst = []
    reward_train_lst = []
    next_obs_train_lst = []
    terminal_train_lst = []
    task_train_lst = []

    for train_path in train_paths:
        task_idx = extract_task_idx(train_path)
        trj_npy = np.load(train_path, allow_pickle=True)
        obs_train_lst += list(trj_npy[:, 0])
        action_train_lst += list(trj_npy[:, 1])
        reward_train_lst += list(trj_npy[:, 2])
        next_obs_train_lst += list(trj_npy[:, 3])
        terminal = [0 for _ in range(trj_npy.shape[0])]
        terminal[-1] = 1
        terminal_train_lst += terminal
        task_train_lst += [task_idx for _ in range(trj_npy.shape[0])]

    obs_train_np = np.asarray(obs_train_lst)
    next_obs_train_np = np.asarray(next_obs_train_lst)
    obs_normalizer.update(obs_train_np)
    env.update_obs_mean_var(obs_normalizer.mean, obs_normalizer.var)
    obs_train_np = obs_normalizer.forward(obs_train_np)
    next_obs_train_np = obs_normalizer.forward(next_obs_train_np)

    for task_train, obs, action, reward, next_obs, terminal in zip(
        task_train_lst,
        obs_train_np,
        action_train_lst,
        reward_train_lst,
        next_obs_train_np,
        terminal_train_lst,
    ):
        train_buffer.add_sample(
            task_train,
            obs,
            action,
            reward,
            terminal,
            next_obs,
            **{"env_info": {}},
        )

    return train_buffer, obs_normalizer


def build_context_encoder_decoder(variant, obs_dim: int, action_dim: int) -> Tuple[MlpEncoder, MlpDecoder]:
    latent_dim = variant["latent_size"]
    net_size = variant["net_size"]
    use_next_obs_in_context = variant["algo_params"]["use_next_obs_in_context"]
    use_information_bottleneck = variant["algo_params"]["use_information_bottleneck"]

    context_encoder_input_dim = (
        2 * obs_dim + action_dim + 1
        if use_next_obs_in_context
        else obs_dim + action_dim + 1
    )
    context_encoder_output_dim = latent_dim * 2 if use_information_bottleneck else latent_dim

    context_encoder = MlpEncoder(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        output_activation=torch.tanh,
        batch_attention=False,
    ).to(ptu.device)

    context_decoder = MlpDecoder(
        hidden_size=net_size,
        num_hidden_layers=3,
        z_dim=latent_dim,
        action_dim=action_dim,
        obs_dim=obs_dim,
        reward_dim=1,
        use_next_obs_in_context=use_next_obs_in_context,
        ensemble_size=1,
    ).to(ptu.device)

    return context_encoder, context_decoder


def build_task_dynamics(variant, obs_dim: int, action_dim: int) -> MultiTaskDynamics:
    algo_params = variant["algo_params"]
    use_next_obs_in_context = algo_params["use_next_obs_in_context"]
    if use_next_obs_in_context:
        num_hidden_layers = 3
        dynamics_weight_decay = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5]
    else:
        num_hidden_layers = 2
        dynamics_weight_decay = [2.5e-5, 5e-5, 7.5e-5]

    task_dynamics = MultiTaskDynamics(
        num_tasks=variant["n_train_tasks"],
        hidden_size=variant["net_size"],
        num_hidden_layers=num_hidden_layers,
        action_dim=action_dim,
        obs_dim=obs_dim,
        reward_dim=1,
        use_next_obs_in_context=use_next_obs_in_context,
        ensemble_size=algo_params["ensemble_size"],
        dynamics_weight_decay=dynamics_weight_decay,
    )
    task_dynamics.to(ptu.device)
    return task_dynamics


def load_task_dynamics(task_dynamics: MultiTaskDynamics, variant, seed) -> Path:
    dynamics_dir = Path(__file__).parent.absolute() / "dynamics" / variant["env_name"] / f"expert_seed{seed}"
    try:
        task_dynamics.load(str(dynamics_dir))
    except (FileNotFoundError, RuntimeError) as error:
        raise RuntimeError(
            f"Failed to load pretrained task dynamics from {dynamics_dir}. "
            "Run pretrain_dynamics.py with the same config before relabel pretraining, "
            "or set algo_params.relabel_data_ratio=0 to disable relabel data."
        ) from error

    for model in task_dynamics.models:
        model.eval()
    return dynamics_dir


def sample_context_batch(
    train_buffer: MultiTaskReplayBuffer,
    task_indices: Sequence[int],
    batch_size: int,
):
    obs_batch = []
    action_batch = []
    reward_batch = []
    next_obs_batch = []
    terms_batch = []

    for task_idx in task_indices:
        batch = train_buffer.random_batch(task_idx, batch_size)
        obs_batch.append(ptu.from_numpy(batch["observations"]))
        action_batch.append(ptu.from_numpy(batch["actions"]))
        reward_batch.append(ptu.from_numpy(batch["rewards"]))
        next_obs_batch.append(ptu.from_numpy(batch["next_observations"]))
        terms_batch.append(ptu.from_numpy(batch["terminals"]).float())

    obs_batch = torch.stack(obs_batch, dim=0)
    action_batch = torch.stack(action_batch, dim=0)
    reward_batch = torch.stack(reward_batch, dim=0)
    next_obs_batch = torch.stack(next_obs_batch, dim=0)
    terms_batch = torch.stack(terms_batch, dim=0)
    return obs_batch, action_batch, reward_batch, next_obs_batch, terms_batch


def build_context(obs, actions, rewards, next_obs, use_next_obs_in_context: bool) -> torch.Tensor:
    if use_next_obs_in_context:
        return torch.cat([obs, actions, rewards, next_obs], dim=-1)
    return torch.cat([obs, actions, rewards], dim=-1)


def sample_real_context(
    train_buffer: MultiTaskReplayBuffer,
    task_indices: Sequence[int],
    batch_size: int,
    use_next_obs_in_context: bool,
) -> torch.Tensor:
    obs, actions, rewards, next_obs, _terms = sample_context_batch(
        train_buffer=train_buffer,
        task_indices=task_indices,
        batch_size=batch_size,
    )
    return build_context(obs, actions, rewards, next_obs, use_next_obs_in_context)


def _sort_context_by_uncertainty(context: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
    sorted_indices = torch.argsort(uncertainty, dim=-1)
    gather_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, context.size(-1))
    return torch.gather(context, dim=1, index=gather_indices)


@torch.no_grad()
def make_relabel_buffer(
    train_buffer: MultiTaskReplayBuffer,
    relabel_buffer: MultiTaskContextBuffer,
    task_dynamics: MultiTaskDynamics,
    train_tasks: Sequence[int],
    context_batch_size: int,
    relabel_sample_multiplier: int,
    num_aug_neg_tasks: int,
    obs_dim: int,
    action_dim: int,
    use_next_obs_in_context: bool,
) -> int:
    relabel_buffer.clear(train_tasks)
    sample_batch_size = max(1, int(context_batch_size) * max(1, int(relabel_sample_multiplier)))
    task_indices = np.asarray(train_tasks)

    context_batch = sample_real_context(
        train_buffer=train_buffer,
        task_indices=task_indices,
        batch_size=sample_batch_size,
        use_next_obs_in_context=use_next_obs_in_context,
    )
    c_mb, c_b, _ = context_batch.shape
    relabel_context = context_batch.clone()
    relabel_obs = relabel_context[:, :, :obs_dim].reshape(c_mb * c_b, -1)

    relabel_actions = relabel_context[:, :, obs_dim:obs_dim + action_dim].reshape(c_mb * c_b, -1)

    relabel_output, relabel_std = task_dynamics.step(
        relabel_obs,
        relabel_actions,
        task_indices=task_indices,
        return_std=True,
    )
    relabel_context[:, :, obs_dim + action_dim:] = relabel_output.reshape(c_mb, c_b, -1)
    sorted_relabel = _sort_context_by_uncertainty(relabel_context, relabel_std)
    relabel_buffer.add_sample(task_indices, ptu.get_numpy(sorted_relabel))
    num_added = c_mb * c_b

    max_neg_tasks = max(0, len(task_indices) - 1)
    num_aug = min(max(0, int(num_aug_neg_tasks)), max_neg_tasks)
    if num_aug <= 0:
        return num_added

    all_neg_indices = np.asarray([
        np.random.choice(task_indices[task_indices != task_idx], num_aug, replace=False)
        for task_idx in task_indices
    ])
    for aug_idx in range(num_aug):
        neg_indices = all_neg_indices[:, aug_idx]
        relabel_output, relabel_std = task_dynamics.step(
            relabel_obs,
            relabel_actions,
            task_indices=neg_indices,
            return_std=True,
        )
        relabel_context[:, :, obs_dim + action_dim:] = relabel_output.reshape(c_mb, c_b, -1)
        sorted_relabel = _sort_context_by_uncertainty(relabel_context, relabel_std)
        relabel_buffer.add_sample(neg_indices, ptu.get_numpy(sorted_relabel))
        num_added += c_mb * c_b

    return num_added


def sample_pretrain_context(
    train_buffer: MultiTaskReplayBuffer,
    relabel_buffer: MultiTaskContextBuffer,
    task_indices: Sequence[int],
    context_batch_size: int,
    relabel_data_ratio: float,
    use_next_obs_in_context: bool,
) -> torch.Tensor:
    relabel_data_size = 0 if relabel_buffer is None else int(context_batch_size * relabel_data_ratio)
    relabel_data_size = min(max(0, relabel_data_size), context_batch_size)
    real_data_size = context_batch_size - relabel_data_size

    contexts = []
    if real_data_size > 0:
        contexts.append(sample_real_context(train_buffer, task_indices, real_data_size, use_next_obs_in_context))
    if relabel_data_size > 0:
        contexts.append(ptu.from_numpy(relabel_buffer.random_batch_task(relabel_data_size, task_indices)))
    if len(contexts) == 0:
        raise ValueError("context_batch_size must be positive")
    if len(contexts) == 1:
        return contexts[0]
    return torch.cat(contexts, dim=1)


def _product_of_gaussians(mus: torch.Tensor, sigmas_squared: torch.Tensor):
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1.0 / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def infer_task_embedding(
    context_encoder: MlpEncoder,
    context: torch.Tensor,
    latent_dim: int,
    use_information_bottleneck: bool,
) -> torch.Tensor:
    params = context_encoder(context)
    params = params.view(context.size(0), -1, context_encoder.output_size)

    if use_information_bottleneck:
        mu = params[..., :latent_dim]
        sigma_squared = F.softplus(params[..., latent_dim:])
        z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        z_means = torch.stack([z_param[0] for z_param in z_params])
        return z_means

    return torch.mean(params, dim=1)


def reconstruction_loss(
    context_decoder: MlpDecoder,
    task_embedding: torch.Tensor,
    context: torch.Tensor,
    obs_dim: int,
    action_dim: int,
) -> torch.Tensor:
    _, context_batch_size, _ = context.size()
    targets = context[..., obs_dim + action_dim:]
    repeated_task_embedding = task_embedding.unsqueeze(1).expand(-1, context_batch_size, -1)
    predictions = context_decoder(
        context[..., :obs_dim],
        context[..., obs_dim:obs_dim + action_dim],
        repeated_task_embedding,
    )
    return torch.mean((targets - predictions) ** 2)


def resolve_pretrain_hparams(variant):
    algo_params = variant["algo_params"]
    return dict(
        num_iters=int(variant.get("num_iters", algo_params.get("num_iterations", 500))),
        decoder_iter=int(variant.get("decoder_iter", 1)),
        encoder_lr=float(variant.get("encoder_lr", algo_params.get("context_lr", 3e-4))),
        recon_loss_weight=float(algo_params.get("recon_loss_weight", 1.0)),
        relabel_data_ratio=float(algo_params.get("relabel_data_ratio", 0.0)),
        relabel_buffer_size=int(algo_params.get("relabel_buffer_size", 50000)),
        relabel_sample_multiplier=int(variant.get("pretrain_relabel_sample_multiplier", 10)),
        relabel_refresh_interval=int(variant.get("pretrain_relabel_refresh_interval", 1)),
        num_aug_neg_tasks=int(algo_params.get("num_aug_neg_tasks", 0)),
        log_interval=int(variant.get("pretrain_log_interval", 100)),
        context_batch_size=int(algo_params.get("embedding_batch_size", 256)),
        meta_batch=int(algo_params.get("meta_batch", variant["n_train_tasks"])),
    )


def experiment(variant, seed=None):
    env = NormalizedBoxEnv(ENVS[variant["env_name"]](**variant["env_params"]))

    if seed is not None:
        global_seed(seed)
        env.seed(seed)

    tasks = env.get_all_task_idx()
    train_tasks = list(tasks[:variant["n_train_tasks"]])
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    ptu.set_gpu_mode(variant["util_params"]["use_gpu"], variant["util_params"]["gpu_id"])
    os.environ["DEBUG"] = str(int(variant["util_params"]["debug"]))

    use_next_obs_in_context = variant["algo_params"]["use_next_obs_in_context"]
    use_information_bottleneck = variant["algo_params"]["use_information_bottleneck"]
    latent_dim = variant["latent_size"]
    context_dim = 2 * obs_dim + action_dim + 1 if use_next_obs_in_context else obs_dim + action_dim + 1
    variant["algo_params"]["context_dim"] = context_dim

    train_buffer, obs_normalizer = build_train_buffer(variant, env, train_tasks, obs_dim)
    context_encoder, context_decoder = build_context_encoder_decoder(variant, obs_dim, action_dim)

    hparams = resolve_pretrain_hparams(variant)
    if not 0.0 <= hparams["relabel_data_ratio"] <= 1.0:
        raise ValueError("relabel_data_ratio must be in [0, 1]")
    hparams["relabel_sample_multiplier"] = max(1, hparams["relabel_sample_multiplier"])
    hparams["relabel_refresh_interval"] = max(1, hparams["relabel_refresh_interval"])
    relabel_enabled = hparams["relabel_data_ratio"] > 0.0

    task_dynamics = None
    relabel_buffer = None
    dynamics_dir = None
    effective_num_aug_neg_tasks = 0
    if relabel_enabled:
        task_dynamics = build_task_dynamics(variant, obs_dim, action_dim)
        dynamics_dir = load_task_dynamics(task_dynamics, variant, seed)
        relabel_buffer = MultiTaskContextBuffer(
            hparams["relabel_buffer_size"],
            env,
            train_tasks,
            context_dim,
        )
        effective_num_aug_neg_tasks = min(hparams["num_aug_neg_tasks"], max(0, len(train_tasks) - 1))
        if effective_num_aug_neg_tasks != hparams["num_aug_neg_tasks"]:
            print(
                f"num_aug_neg_tasks={hparams['num_aug_neg_tasks']} exceeds available negative tasks; "
                f"using {effective_num_aug_neg_tasks}."
            )

    optimizer = torch.optim.Adam(
        [{"params": context_encoder.parameters()}, {"params": context_decoder.parameters()}],
        lr=hparams["encoder_lr"],
    )

    context_encoder.train(True)
    context_decoder.train(True)

    for step1 in range(hparams["num_iters"]):
        if relabel_enabled and step1 % hparams["relabel_refresh_interval"] == 0:
            num_relabel_contexts = make_relabel_buffer(
                train_buffer=train_buffer,
                relabel_buffer=relabel_buffer,
                task_dynamics=task_dynamics,
                train_tasks=train_tasks,
                context_batch_size=hparams["context_batch_size"],
                relabel_sample_multiplier=hparams["relabel_sample_multiplier"],
                num_aug_neg_tasks=effective_num_aug_neg_tasks,
                obs_dim=obs_dim,
                action_dim=action_dim,
                use_next_obs_in_context=use_next_obs_in_context,
            )
        else:
            num_relabel_contexts = 0

        replace = len(train_tasks) < hparams["meta_batch"]
        sampled_task_indices = np.random.choice(train_tasks, size=hparams["meta_batch"], replace=replace)

        for step2 in range(hparams["decoder_iter"]):
            context = sample_pretrain_context(
                train_buffer=train_buffer,
                relabel_buffer=relabel_buffer,
                task_indices=sampled_task_indices,
                context_batch_size=hparams["context_batch_size"],
                relabel_data_ratio=hparams["relabel_data_ratio"],
                use_next_obs_in_context=use_next_obs_in_context,
            )
            task_embedding = infer_task_embedding(
                context_encoder=context_encoder,
                context=context,
                latent_dim=latent_dim,
                use_information_bottleneck=use_information_bottleneck,
            )

            recon_loss_value = reconstruction_loss(
                context_decoder=context_decoder,
                task_embedding=task_embedding,
                context=context,
                obs_dim=obs_dim,
                action_dim=action_dim,
            )
            context_loss = hparams["recon_loss_weight"] * recon_loss_value

            optimizer.zero_grad()
            context_loss.backward()
            optimizer.step()

            global_step = step1 * hparams["decoder_iter"] + step2 + 1
            if global_step == 1 or global_step % hparams["log_interval"] == 0:
                print(
                    f"[Step {global_step}] "
                    f"context_loss={context_loss.item():.6f}, "
                    f"recon_loss={recon_loss_value.item():.6f}, "
                    f"context_batch={hparams['context_batch_size']}, "
                    f"relabel_ratio={hparams['relabel_data_ratio']:.3f}, "
                    f"new_relabel_contexts={num_relabel_contexts}"
                )

    context_encoder.train(False)
    context_decoder.train(False)

    save_dir = Path(__file__).parent.absolute() / "encoder_decoder" / variant["env_name"] / f"expert_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    context_encoder.save(save_dir / "context_encoder.pth")
    context_decoder.save(save_dir / "context_decoder.pth")
    np.savez(save_dir / "obs_normalizer.npz", mean=obs_normalizer.mean, var=obs_normalizer.var)

    save_metadata = {
        "seed": seed,
        "env_name": variant["env_name"],
        "n_train_tasks": variant["n_train_tasks"],
        "latent_size": variant["latent_size"],
        "use_next_obs_in_context": use_next_obs_in_context,
        "use_information_bottleneck": use_information_bottleneck,
        "num_iters": hparams["num_iters"],
        "decoder_iter": hparams["decoder_iter"],
        "encoder_lr": hparams["encoder_lr"],
        "recon_loss_weight": hparams["recon_loss_weight"],
        "context_batch_size": hparams["context_batch_size"],
        "meta_batch": hparams["meta_batch"],
        "relabel_data_ratio": hparams["relabel_data_ratio"],
        "relabel_enabled": relabel_enabled,
        "relabel_action_source": "offline_action",
        "relabel_buffer_size": hparams["relabel_buffer_size"],
        "relabel_sample_multiplier": hparams["relabel_sample_multiplier"],
        "relabel_refresh_interval": hparams["relabel_refresh_interval"],
        "num_aug_neg_tasks": effective_num_aug_neg_tasks,
        "dynamics_dir": None if dynamics_dir is None else str(dynamics_dir),
    }
    with open(save_dir / "pretrain_metadata.json", "w", encoding="utf-8") as file:
        json.dump(save_metadata, file, indent=2)

    print(f"Saved pretrained context encoder and decoder to {save_dir}")


@click.command()
@click.argument("config", default=None)
@click.option("--gpu", default=0)
@click.option("--seed_list", multiple=True, type=int, default=[0, 1, 2, 3])
def main(config, gpu, seed_list):
    variant = default_config
    if config:
        with open(os.path.join(config), encoding="utf-8") as file:
            exp_params = json.load(file)
        variant = deep_update_dict(exp_params, variant)
    variant["util_params"]["gpu_id"] = gpu

    if len(seed_list) > 1:
        with mp.Pool(processes=len(seed_list)) as pool:
            pool.starmap(experiment, product([variant], seed_list))
            pool.close()
            pool.join()
    else:
        experiment(variant, seed=seed_list[0])


if __name__ == "__main__":
    main()
