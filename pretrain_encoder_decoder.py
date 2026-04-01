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
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.autoencoder import MlpDecoder, MlpEncoder
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


def metric_loss(z: torch.Tensor, tasks: Sequence[int], epsilon: float = 1e-3) -> torch.Tensor:
    pos_z_loss = z.new_tensor(0.0)
    neg_z_loss = z.new_tensor(0.0)
    pos_cnt = 0
    neg_cnt = 0
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            if tasks[i] == tasks[j]:
                pos_z_loss = pos_z_loss + torch.sqrt(torch.mean((z[i] - z[j]) ** 2) + epsilon)
                pos_cnt += 1
            else:
                neg_z_loss = neg_z_loss + 1 / (torch.mean((z[i] - z[j]) ** 2) + epsilon * 100)
                neg_cnt += 1
    return pos_z_loss / (pos_cnt + epsilon) + neg_z_loss / (neg_cnt + epsilon)


def supervised_loss(
    context_decoder: MlpDecoder,
    task_embedding: torch.Tensor,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    use_next_obs_in_context: bool,
) -> torch.Tensor:
    repeated_task_embedding = task_embedding.unsqueeze(1).expand(-1, obs.size(1), -1)
    predictions = context_decoder(obs, actions, repeated_task_embedding)
    if use_next_obs_in_context:
        targets = torch.cat([rewards, next_obs], dim=-1)
    else:
        targets = rewards
    return torch.mean((targets - predictions) ** 2)


def resolve_pretrain_hparams(variant):
    algo_params = variant["algo_params"]
    return dict(
        num_iters=int(variant.get("num_iters", algo_params.get("num_iterations", 500))),
        decoder_iter=int(variant.get("decoder_iter", 1)),
        encoder_lr=float(variant.get("encoder_lr", algo_params.get("context_lr", 3e-4))),
        beta_encoder=float(variant.get("beta_encoder", 1.0)),
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

    train_buffer, obs_normalizer = build_train_buffer(variant, env, train_tasks, obs_dim)
    context_encoder, context_decoder = build_context_encoder_decoder(variant, obs_dim, action_dim)

    hparams = resolve_pretrain_hparams(variant)
    optimizer = torch.optim.Adam(
        [{"params": context_encoder.parameters()}, {"params": context_decoder.parameters()}],
        lr=hparams["encoder_lr"],
    )

    use_next_obs_in_context = variant["algo_params"]["use_next_obs_in_context"]
    use_information_bottleneck = variant["algo_params"]["use_information_bottleneck"]
    latent_dim = variant["latent_size"]

    context_encoder.train(True)
    context_decoder.train(True)

    for step1 in range(hparams["num_iters"]):
        sampled_task_indices = np.random.choice(train_tasks, size=hparams["meta_batch"], replace=True)

        for step2 in range(hparams["decoder_iter"]):
            obs, actions, rewards, next_obs, terms = sample_context_batch(
                train_buffer=train_buffer,
                task_indices=sampled_task_indices,
                batch_size=hparams["context_batch_size"],
            )
            context = build_context(obs, actions, rewards, next_obs, use_next_obs_in_context)
            task_embedding = infer_task_embedding(
                context_encoder=context_encoder,
                context=context,
                latent_dim=latent_dim,
                use_information_bottleneck=use_information_bottleneck,
            )

            metric_loss_value = metric_loss(task_embedding, sampled_task_indices)
            supervised_loss_value = supervised_loss(
                context_decoder=context_decoder,
                task_embedding=task_embedding,
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                use_next_obs_in_context=use_next_obs_in_context,
            )
            total_loss = hparams["beta_encoder"] * metric_loss_value + supervised_loss_value

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            global_step = step1 * hparams["decoder_iter"] + step2 + 1
            if global_step == 1 or global_step % hparams["log_interval"] == 0:
                print(
                    f"[Step {global_step}] "
                    f"total_loss={total_loss.item():.6f}, "
                    f"metric_loss={metric_loss_value.item():.6f}, "
                    f"supervised_loss={supervised_loss_value.item():.6f}, "
                    f"context_batch={hparams['context_batch_size']}"
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
        "beta_encoder": hparams["beta_encoder"],
        "context_batch_size": hparams["context_batch_size"],
        "meta_batch": hparams["meta_batch"],
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
