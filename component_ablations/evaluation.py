"""Main evaluation and read-only, shared fixed-context diagnostics."""
from pathlib import Path
import numpy as np
import torch
from .common import digest_json, seed_for, verify_file, write_json, read_json
from .randomness import isolated_rng


def rollout(algorithm, task, reset_seed, horizon=200):
    env = algorithm.env
    env.seed(reset_seed)
    env.reset_task(task)
    obs = env.reset()
    total, length = 0.0, 0
    for _ in range(horizon):
        action, _ = algorithm.agent.get_action(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total += float(reward)
        length += 1
        if done:
            break
    if not np.isfinite(total):
        raise FloatingPointError("Nonfinite evaluation return")
    return total, length


def main_evaluation(algorithm, config, output, seed, iteration, smoke=False):
    """200-row pool; sample 200 per full episode; cumulative sampled context.

    Evaluation transitions never update context. Initial collection returns are
    excluded. Paired reset/context seeds exclude arm and training seed.
    """
    output = Path(output)
    env_name = config["env_name"]
    records = []
    tasks = list(algorithm.train_tasks[:1]) + list(algorithm.eval_tasks[:1]) if smoke else list(algorithm.train_tasks) + list(algorithm.eval_tasks)
    device = next(algorithm.agent.policy.parameters()).device
    for task in tasks:
        split = "seen" if task in algorithm.train_tasks else "unseen"
        for protocol in ("main_given_context", "main_self_collected"):
            with isolated_rng("component-main-v1", env_name, task, protocol, iteration, "collection"):
                if protocol == "main_self_collected":
                    algorithm.env.seed(seed_for("component-main-v1", env_name, task, "collection_reset"))
                    algorithm.context_buffer.clear()
                    algorithm.collect_context_by_expl_policy(task, given_sample_num=200)
                    if algorithm.context_buffer._size != 200:
                        raise ValueError("Self-collected pool must contain exactly 200 transitions")
                    pool = algorithm.context_buffer
                else:
                    pool = (algorithm.train_buffer if split == "seen" else algorithm.eval_buffer).task_buffers[task]
                for repeat in range(1 if smoke else algorithm.num_evals):
                    algorithm.agent.clear_z()
                    cumulative = []
                    steps, episode = 0, 0
                    while steps < (200 if smoke else 600):
                        # No eval rollout rows are appended to this context.
                        row_seed = seed_for("component-main-v1", env_name, task, protocol, repeat, episode, "rows")
                        rows = np.random.RandomState(row_seed).randint(0, pool._size, 200)
                        batch = pool.sample_data(rows)
                        cumulative.append(algorithm.unpack_context_batch(batch))
                        context = torch.as_tensor(np.concatenate(cumulative, axis=1), dtype=torch.float32, device=device)
                        with torch.no_grad():
                            algorithm.agent.infer_posterior(context)
                        reset_seed = seed_for("component-main-v1", env_name, task, "shared_reset", repeat, episode)
                        score, length = rollout(algorithm, task, reset_seed)
                        records.append({"protocol": protocol, "training_seed": seed, "iteration": iteration,
                                        "task_id": task, "split": split, "repeat": repeat, "episode": episode,
                                        "return": score, "length": length, "reset_seed": reset_seed,
                                        "context_row_seed": row_seed, "context_rows_sha256": digest_json(rows.tolist()),
                                        "cumulative_context_length": context.shape[1], "initial_pool_rows": 200 if protocol == "main_self_collected" else pool._size})
                        steps += length
                        episode += 1
                        if episode > 10000:
                            raise RuntimeError("Evaluation failed to make progress")
    write_json(output / ("main_itr_%d.json" % iteration), {"kind": "smoke_only" if smoke else "formal_main",
                                                            "iteration": iteration, "records": records})
    return records


def validate_bank(manifest, root=None):
    if manifest.get("kind") != "formal" or manifest.get("complete") is not True:
        raise ValueError("A complete FORMAL bank is required; shift smoke banks are rejected")
    if manifest.get("protocol") != "shift-diagnostics-v3":
        raise ValueError("Bind the latest frozen v3 protocol explicitly; do not downgrade automatically")
    if manifest.get("collector", {}).get("method") != "GENTLE":
        raise ValueError("Fixed diagnostic collector must be original GENTLE")
    if manifest.get("observation_space") != "raw" or manifest.get("action_space") != "normalized":
        raise ValueError("Bank coordinates must be raw observations and normalized actions")
    conditions = {"offline", "policy499_sigma000", "policy499_sigma020"}
    contexts = [item for item in manifest["contexts"] if item["condition_id"] in conditions]
    keys = {(item["task_id"], item["condition_id"], item["context_repeat"]) for item in contexts}
    expected = {(task, condition, k) for task in range(20) for condition in conditions for k in range(20)}
    if keys != expected or len(contexts) != len(expected) or any(item["context_length"] != 200 for item in contexts):
        raise ValueError("Formal bank coverage must be 20 tasks x 3 conditions x 20 contexts of length 200")
    if len({item["context_id"] for item in contexts}) != len(contexts):
        raise ValueError("Duplicate context identities in formal bank")
    if any(item["task_split"] != ("seen" if item["task_id"] < 10 else "unseen") for item in contexts):
        raise ValueError("Fixed-context task split labels are inconsistent")
    if root is not None:
        verify_file(manifest["arrays"], root)
    return contexts


def representation_metrics(embeddings, metadata, tasks):
    conditions = ("offline", "policy499_sigma000", "policy499_sigma020")
    lookup = {(item["task_id"], item["condition_id"], item["context_repeat"]): embeddings[i] for i, item in enumerate(metadata)}
    result = []
    for reference in ("offline", "policy499_sigma000"):
        reference_points = {task: np.stack([lookup[task, reference, k] for k in range(20)]) for task in tasks}
        centers = np.stack([reference_points[task].mean(0) for task in tasks])
        distances = ((centers[:, None] - centers[None]) ** 2).sum(-1)
        separation = float(distances[~np.eye(len(tasks), dtype=bool)].mean())
        classifier_centers = np.stack([reference_points[task][:10].mean(0) for task in tasks])
        for condition in conditions:
            points = np.stack([[lookup[task, condition, k] for k in range(20)] for task in tasks])
            means = points.mean(1)
            drift = float(((means - centers) ** 2).sum(-1).mean())
            within = float(((points - means[:, None]) ** 2).sum(-1).mean())
            predictions = ((points[:, 10:, None] - classifier_centers[None, None]) ** 2).sum(-1).argmin(-1)
            accuracy = float((predictions == np.arange(len(tasks))[:, None]).mean())
            collapse = not np.isfinite(separation) or separation <= 1e-12
            result.append({"reference": reference, "condition": condition, "B": drift, "S": separation,
                           "B_over_S": None if collapse else drift / separation, "collapse": collapse,
                           "collapse_threshold": 1e-12, "W": within, "identification_accuracy": accuracy})
    return result


def fixed_evaluation(algorithm, config, binding, root, output, seed):
    manifest = read_json(verify_file(binding, root))
    resources = read_json(Path(root) / "resources.json")[config["env_name"]]
    if manifest.get("environment") != config["env_name"] or manifest.get("task_manifest_sha256") != resources["historical_task_manifest"]["sha256"]:
        raise ValueError("Fixed bank task/environment binding mismatch")
    contexts = validate_bank(manifest, Path(binding["path"]).parent if Path(binding["path"]).is_absolute() else Path(root) / Path(binding["path"]).parent)
    manifest_dir = verify_file(binding, root).parent
    arrays = np.load(str(verify_file(manifest["arrays"], manifest_dir)), allow_pickle=False)
    device = next(algorithm.agent.policy.parameters()).device
    latents, records, metadata = [], [], []
    index_by_id = {item["context_id"]: i for i, item in enumerate(manifest["contexts"])}
    for item in contexts:
        prefix = item.get("array_prefix", "c%d_" % index_by_id[item["context_id"]])
        obs, action, reward, next_obs = [arrays[prefix + name] for name in ("observations", "actions", "rewards", "next_observations")]
        fields = [algorithm.obs_normalizer.forward(obs), action, reward]
        if algorithm.use_next_obs_in_context:
            fields.append(algorithm.obs_normalizer.forward(next_obs))
        context = torch.as_tensor(np.concatenate(fields, -1)[None], dtype=torch.float32, device=device)
        with torch.no_grad(), isolated_rng("component-fixed", seed, item["context_id"], "posterior"):
            algorithm.agent.clear_z()
            z = algorithm._get_context_embedding(context, sample=False)
            algorithm.agent.infer_posterior(context)
        latents.append(z[0].cpu().tolist())
        metadata.append(item)
        for repeat in range(3):
            reset_seed = seed_for("shift-diagnostics-v3", config["env_name"], item["task_id"], "evaluation_reset", "shared", item["context_repeat"], repeat)
            score, length = rollout(algorithm, item["task_id"], reset_seed)
            records.append({"task_id": item["task_id"], "split": item["task_split"], "condition": item["condition_id"],
                            "context_id": item["context_id"], "context_repeat": item["context_repeat"],
                            "rollout": repeat, "reset_seed": reset_seed, "return": score, "length": length, "seed": seed})
    output = Path(output)
    write_json(output / "fixed_context.json", {"kind": "formal_fixed", "bank": binding, "records": records})
    np.savez_compressed(str(output / "embeddings.npz"), latent=np.asarray(latents))
    write_json(output / "embedding_metadata.json", metadata)
    write_json(output / "representation_metrics.json", {split: representation_metrics(np.asarray(latents), metadata, list(tasks))
                                                         for split, tasks in (("seen", algorithm.train_tasks), ("unseen", algorithm.eval_tasks))})
