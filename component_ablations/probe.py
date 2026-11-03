"""Frozen-encoder, seen-only feedback probe; held-out queries never select models."""
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from .common import read_json, verify_file, seed_for, write_json
from .randomness import isolated_rng
from .state import atomic_torch_save


def validate_probe_manifest(manifest):
    if manifest.get("kind") != "formal_probe" or manifest.get("complete") is not True:
        raise ValueError("Probe requires a complete formal_probe manifest")
    groups = manifest["groups"]
    keys = {(g["task_id"], g["split"]) for g in groups}
    expected = {(i, s) for i in range(10) for s in ("train", "val", "test")} | {(i, "test") for i in range(10, 20)}
    if keys != expected or len(groups) != len(expected):
        raise ValueError("Probe must cover seen train/val/test and unseen test only")
    used = {}
    for group in groups:
        task = group["task_id"]
        if len(group["query_trajectory_ids"]) != 1000 or len(group["context_trajectory_ids"]) != 20:
            raise ValueError("Probe requires 1000 queries and 20 independent contexts per task/split")
        context_sets = [set(item) if isinstance(item, list) else {item} for item in group["context_trajectory_ids"]]
        c = set()
        for trajectories in context_sets:
            if not trajectories or c & trajectories:
                raise ValueError("Probe contexts must come from independent trajectories")
            c.update(trajectories)
        q = set(group["query_trajectory_ids"])
        if q & c:
            raise ValueError("Probe context/query trajectories overlap")
        seen_before = used.setdefault(task, set())
        if seen_before & (q | c):
            raise ValueError("Probe trajectories overlap across train/val/test")
        seen_before.update(q | c)
    return groups


def run_probe(algorithm, config, binding, root, output, training_seed):
    if not binding:
        raise ValueError("No frozen probe manifest is bound; do not fabricate query data")
    path = verify_file(binding, root)
    manifest = read_json(path)
    groups = validate_probe_manifest(manifest)
    if manifest["environment"] != config["env_name"]:
        raise ValueError("Probe environment mismatch")
    resources = read_json(Path(root) / "resources.json")[config["env_name"]]
    if manifest.get("task_manifest_sha256") != resources["historical_task_manifest"]["sha256"]:
        raise ValueError("Probe task manifest mismatch")
    arrays = np.load(str(verify_file(manifest["arrays"], path.parent)), allow_pickle=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = next(algorithm.agent.policy.parameters()).device
    prepared = {}
    with torch.no_grad():
        for group in groups:
            task, split = group["task_id"], group["split"]
            qp, cp = group["query_prefix"], group["context_prefix"]
            fields = [arrays[cp + name] for name in ("observations", "actions", "rewards", "next_observations")]
            if any(array.shape[:2] != (20, 200) for array in fields):
                raise ValueError("Probe context array shape mismatch")
            context_fields = [algorithm.obs_normalizer.forward(fields[0]), fields[1], fields[2]]
            if algorithm.use_next_obs_in_context:
                context_fields.append(algorithm.obs_normalizer.forward(fields[3]))
            latent = algorithm._get_context_embedding(torch.as_tensor(np.concatenate(context_fields, -1), dtype=torch.float32, device=device), sample=False).detach()
            obs, action, reward, next_obs = [arrays[qp + name] for name in ("observations", "actions", "rewards", "next_observations")]
            if any(len(array) != 1000 for array in (obs, action, reward, next_obs)):
                raise ValueError("Probe query array shape mismatch")
            conditions = arrays[qp + "conditions"].tolist()
            if conditions.count("policy499_sigma000") != 500 or conditions.count("policy499_sigma020") != 500:
                raise ValueError("Probe query mixture must be 500 sigma000 + 500 sigma020")
            inputs = np.concatenate([algorithm.obs_normalizer.forward(obs), action], -1)
            target = np.concatenate([reward.reshape(1000, 1), next_obs - obs], -1)
            prepared[task, split] = (torch.as_tensor(inputs, dtype=torch.float32, device=device), latent,
                                      torch.as_tensor(target, dtype=torch.float32, device=device))
    train_targets = torch.cat([prepared[i, "train"][2] for i in range(10)])
    target_mean = train_targets.mean(0)
    target_std = train_targets.std(0, unbiased=False).clamp(min=1e-6)
    train_z = torch.cat([prepared[i, "train"][1] for i in range(10)])
    z_mean = train_z.mean(0)
    z_std = train_z.std(0, unbiased=False).clamp(min=1e-6)

    def evaluate(model, split):
        metrics = []
        for task in range(10 if split == "val" else 20):
            inputs, zs, targets = prepared[task, split]
            error_sum = torch.zeros_like(target_mean)
            standardized_sum = torch.zeros_like(target_mean)
            count = 20 * 1000
            for offset in range(0, count, 4096):
                ids = torch.arange(offset, min(offset + 4096, count), device=device)
                qi, zi = ids % 1000, ids // 1000
                prediction_standard = model(torch.cat([inputs[qi], (zs[zi] - z_mean) / z_std], -1))
                error = prediction_standard * target_std + target_mean - targets[qi]
                error_sum += error.square().sum(0)
                standardized_sum += (error / target_std).square().sum(0)
            mse, nmse = error_sum / count, standardized_sum / count
            metrics.append({"task_id": task, "split": "seen" if task < 10 else "unseen",
                            "reward_MSE": mse[0].item(), "dynamics_MSE": mse[1:].mean().item(),
                            "reward_NMSE": nmse[0].item(), "dynamics_NMSE": nmse[1:].mean().item()})
        return metrics

    with isolated_rng("component-probe", config["env_name"], training_seed):
        rng = np.random.RandomState(seed_for("component-probe-batches", config["env_name"], training_seed))
        first = prepared[0, "train"]
        model = nn.Sequential(nn.Linear(first[0].shape[1] + first[1].shape[1], 128), nn.ReLU(),
                              nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, target_mean.numel())).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
        best, best_update, history = float("inf"), None, []
        for update in range(1, 10001):
            task_ids = rng.randint(0, 10, 256)
            query_ids, context_ids = rng.randint(0, 1000, 256), rng.randint(0, 20, 256)
            inputs = torch.stack([prepared[int(t), "train"][0][int(q)] for t, q in zip(task_ids, query_ids)])
            zs = torch.stack([prepared[int(t), "train"][1][int(c)] for t, c in zip(task_ids, context_ids)])
            targets = torch.stack([prepared[int(t), "train"][2][int(q)] for t, q in zip(task_ids, query_ids)])
            error = model(torch.cat([inputs, (zs - z_mean) / z_std], -1)) - (targets - target_mean) / target_std
            loss = error[:, :1].square().mean() + error[:, 1:].square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite probe training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if update % 500 == 0:
                with torch.no_grad():
                    val_metrics = evaluate(model, "val")
                validation = float(np.mean([m["reward_NMSE"] + m["dynamics_NMSE"] for m in val_metrics]))
                history.append({"update": update, "training_loss": loss.item(), "seen_validation_loss": validation})
                if validation < best:
                    best, best_update = validation, update
                    atomic_torch_save(output / "best.pt", {"state_dict": model.state_dict(), "update": update,
                                                            "target_mean": target_mean, "target_std": target_std,
                                                            "z_mean": z_mean, "z_std": z_std})
        model.load_state_dict(torch.load(str(output / "best.pt"), map_location=device, weights_only=True)["state_dict"])
        with torch.no_grad():
            test_metrics = evaluate(model, "test")
    write_json(output / "training_history.json", history)
    write_json(output / "results.json", {"kind": "formal_probe", "manifest": binding, "training_seed": training_seed,
                                        "selection": "minimum seen-validation; earlier on ties", "selected_update": best_update,
                                        "train_updates": 10000, "metrics": test_metrics,
                                        "target_stats_fit": "seen train queries only", "latent_stats_fit": "seen train contexts only"})
