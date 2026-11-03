"""One subprocess per environment/arm/seed. GPU and engine are bound by parent."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .common import digest_json, now, read_json, sha256, source_identity, verify_file, write_json


def training_resource_digest(entry):
    return digest_json({key: value for key, value in entry.items() if key not in ("diagnostic_bank", "probe_manifest")})


def initialization_identity(root, env, seed, resources):
    return {"environment": env, "seed": seed,
            "base_config_sha256": sha256(root / "reference_configs" / (env + ".json")),
            "resources_digest": training_resource_digest(resources),
            "source_digest": source_identity()["source_digest"], "contract_sha256": sha256(root / "loss_contract.json")}


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def prune_history(path, next_iteration):
    path = Path(path)
    if not path.exists():
        return
    temporary = path.with_name(path.name + ".resume.tmp")
    with path.open("r", encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if source.read().strip():
                    raise ValueError("Corrupt history before trailing interrupted record: " + str(path))
                break
            if row["iteration"] < next_iteration:
                target.write(json.dumps(row) + "\n")
    os.replace(temporary, path)


def output_hashes(run, paths):
    return [{"path": str(Path(path).relative_to(run)).replace("\\", "/"), "sha256": sha256(path)} for path in paths]


def save_archive(path, state):
    """Never overwrite an already published scientific checkpoint."""
    import torch
    from .state import atomic_torch_save
    if path.exists():
        old = torch.load(str(path), map_location="cpu", weights_only=False)
        if old["identity"] != state["identity"] or old["next_iteration"] != state["next_iteration"]:
            raise ValueError("Existing archive identity mismatch: " + str(path))
        for name, network in state["networks"].items():
            for key, tensor in network.items():
                if not torch.equal(tensor.cpu(), old["networks"][name][key].cpu()):
                    raise ValueError("Replayed iteration differs from existing archive; refusing overwrite: " + str(path))
        return
    atomic_torch_save(path, state)


def run_worker(args):
    # No torch/runtime import above this point: visibility and engine selection
    # must be supplied in this process's environment before CUDA initialization.
    import numpy as np
    import torch
    import gtimer as gt
    from .runtime import build_algorithm
    from .state import training_state, load_training_state, atomic_torch_save
    from .randomness import isolated_rng
    from .evaluation import main_evaluation, fixed_evaluation

    root = Path(args.root).resolve()
    protocol, audit = read_json(root / "protocol.json"), read_json(root / "resource_audit.json")
    source = source_identity()
    if not audit["resources_ready"] or audit["resources_sha256"] != sha256(root / "resources.json"):
        raise ValueError("Resources missing/changed: run a successful audit first")
    if source["source_digest"] != audit["source"]["source_digest"]:
        raise ValueError("Source changed after audit; rerun checks and audit")
    entry = read_json(root / "resources.json")[args.env]
    if args.phase in ("fixed", "probe"):
        key = "diagnostic_bank" if args.phase == "fixed" else "probe_manifest"
        if not entry.get(key):
            raise ValueError("Missing formal " + key + "; bind verified resources before launching this phase")
    config_path = root / "configs" / args.env / (args.arm + ".json")
    config = read_json(config_path)
    matrix = read_json(root / "run_matrix.json")["slots"]
    slot = next(row for row in matrix if row["environment"] == args.env and row["arm"] == args.arm and row["seed"] == args.seed)
    if sha256(config_path) != slot["config_sha256"]:
        raise ValueError("Frozen arm configuration changed")
    init_identity = initialization_identity(root, args.env, args.seed, entry)
    initial_path = root / slot["initial_state"]
    if args.phase == "init":
        if initial_path.exists():
            existing = read_json(initial_path.parent / "manifest.json")
            if existing["identity"] != init_identity or existing["sha256"] != sha256(initial_path):
                raise ValueError("Existing immutable initialization differs")
            print("Verified existing initialization", initial_path)
            return
        full_config = read_json(root / "configs" / args.env / "full.json")
        construction_start = time.monotonic()
        algorithm = build_algorithm(full_config, entry, args.seed, root, cpu=args.cpu)
        algorithm.itr = -1
        snapshot = training_state(algorithm, init_identity, 0, include_buffers=False)
        atomic_torch_save(initial_path, snapshot)
        write_json(initial_path.parent / "manifest.json", {"identity": init_identity, "sha256": sha256(initial_path),
                   "created_at": now(), "networks": list(snapshot["networks"]), "optimizer_start": "empty Adam states",
                   "backend": algorithm.backend_identity, "initialization_load_seconds": time.monotonic() - construction_start,
                   "historical_pretraining_seconds": None,
                   "resource_dynamics": entry["pretrained"][str(args.seed)]["dynamics"]}, exclusive=True)
        return

    if not initial_path.exists():
        raise FileNotFoundError("Prepare immutable initial state first: " + str(initial_path))
    init_manifest = read_json(initial_path.parent / "manifest.json")
    if init_manifest["identity"] != init_identity or sha256(initial_path) != init_manifest["sha256"]:
        raise ValueError("Initialization/source/resources mismatch")
    smoke = args.phase == "smoke"
    run = root / (("smoke/%s/%s/seed%d" % (args.env, args.arm, args.seed)) if smoke else slot["run_directory"])
    run.mkdir(parents=True, exist_ok=True)
    if args.phase in ("main", "fixed", "probe"):
        output = run / "evaluation"
        if not (run / "completion.json").exists():
            raise ValueError("Evaluate only completed matched training slots")
    else:
        output = run / "evaluation"
        if (run / "completion.json").exists():
            raise FileExistsError("Completed run is immutable: " + str(run))
        if (run / "resolved_config.json").exists() and not args.resume:
            raise FileExistsError("Run exists; --resume requires the complete checkpoint named in last_checkpoint.json")
    resolved = copy.deepcopy(config)
    if smoke:
        resolved["algo_params"]["num_iterations"] = 1
        resolved["algo_params"]["num_train_steps_per_itr"] = 2
    identity = {**init_identity, "config_sha256": sha256(config_path),
                "resolved_digest": digest_json(resolved), "initial_sha256": init_manifest["sha256"], "kind": "smoke_only" if smoke else "formal"}
    algorithm = build_algorithm(resolved, entry, args.seed, root, cpu=args.cpu)
    algorithm.audit_gradients = smoke
    load_training_state(algorithm, torch.load(str(initial_path), map_location="cpu", weights_only=False), init_identity, require_buffers=False)
    decoder_before = {key: value.detach().cpu().clone() for key, value in algorithm.context_decoder.state_dict().items()}
    if args.phase in ("main", "fixed", "probe"):
        completion = read_json(run / "completion.json")
        checkpoint = run / "checkpoints/itr_499.pt"
        if completion["identity"] != identity or sha256(checkpoint) != completion["final_checkpoint_sha256"]:
            raise ValueError("Completed model identity/hash changed")
        load_training_state(algorithm, torch.load(str(checkpoint), map_location="cpu", weights_only=False), identity, require_buffers=False)
        algorithm.training_mode(False)
        if args.phase == "main":
            main_evaluation(algorithm, resolved, output, args.seed, 499)
            result_paths = [output / "main_itr_499.json"]
        elif args.phase == "fixed":
            if not entry.get("diagnostic_bank"):
                raise ValueError("No formal diagnostic bank bound; smoke banks cannot be substituted")
            fixed_evaluation(algorithm, resolved, entry["diagnostic_bank"], root, output, args.seed)
            result_paths = [output / name for name in ("fixed_context.json", "embeddings.npz", "embedding_metadata.json", "representation_metrics.json")]
        else:
            from .probe import run_probe
            run_probe(algorithm, resolved, entry.get("probe_manifest"), root, run / "probes", args.seed)
            result_paths = [run / "probes" / name for name in ("results.json", "training_history.json", "best.pt")]
        binding_key = "diagnostic_bank" if args.phase == "fixed" else "probe_manifest" if args.phase == "probe" else None
        write_json(run / (args.phase + "_completion.json"), {"identity": identity, "complete": True, "created_at": now(), "checkpoint_sha256": sha256(checkpoint),
                                                           "outputs": output_hashes(run, result_paths), "input_binding": entry[binding_key] if binding_key else None})
        return

    provenance_path = run / "provenance.json"
    if not provenance_path.exists():
        write_json(run / "resolved_config.json", resolved, exclusive=True)
        write_json(provenance_path, {"identity": identity, "created_at": now(), "source": source,
                                   "command": sys.argv, "physical_gpu": args.gpu, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                   "torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
                                   "device": "cpu" if args.cpu else torch.cuda.get_device_name(0),
                                   "mujoco_path": os.environ.get("MUJOCO_PY_MJPRO_PATH"),
                                   "backend": algorithm.backend_identity,
                                   "masks": resolved["component_ablation"], "resources": entry,
                                   "init_manifest": init_manifest,
                                   "rng_rule": "real stream + isolated counter-keyed auxiliary sampling; masks applied after generation",
                                   "historical_full_reused": False}, exclusive=True)
    elif read_json(provenance_path)["identity"] != identity:
        raise ValueError("Run provenance/config mismatch")
    start = 0
    if args.resume:
        last = verify_file(read_json(run / "last_checkpoint.json"), run)
        start = load_training_state(algorithm, torch.load(str(last), map_location="cpu", weights_only=False), identity)
        prune_history(run / "training_steps.jsonl", start)
        prune_history(run / "progress.jsonl", start)
    gt.reset()
    gt.set_def_unique(False)
    wall_start = time.monotonic()
    if not args.cpu:
        torch.cuda.reset_peak_memory_stats()
    checkpoint_iterations = set(protocol["checkpoints"])
    for iteration in range(start, resolved["algo_params"]["num_iterations"]):
        iteration_start = time.monotonic()
        algorithm.itr = iteration
        algorithm.real_batch_digest = hashlib.sha256()
        algorithm.training_mode(True)
        if iteration == 0:
            for task in algorithm.train_tasks:
                algorithm.task_idx = task
                algorithm.env.reset_task(task)
                algorithm.collect_data(algorithm.num_initial_steps, 1, np.inf, buffer=algorithm.train_buffer, add_to_enc_buffer=True)
        for _ in range(algorithm.num_tasks_sample):
            task = np.random.choice(algorithm.train_tasks, 1)[0]
            algorithm.task_idx = task
            algorithm.env.reset_task(task)
            algorithm.enc_replay_buffer.task_buffers[task].clear()
            for count, posterior_rate, add_encoder in ((algorithm.num_steps_prior, np.inf, True),
                                                       (algorithm.num_steps_posterior, algorithm.update_post_train, True),
                                                       (algorithm.num_extra_rl_steps_posterior, algorithm.update_post_train, False)):
                if count:
                    algorithm.collect_data(count, 1, posterior_rate, buffer=algorithm.train_buffer, add_to_enc_buffer=add_encoder)
        for _ in range(algorithm.num_train_steps_per_itr):
            indices = np.random.choice(algorithm.train_tasks, algorithm.meta_batch, replace=len(algorithm.train_tasks) < algorithm.meta_batch)
            algorithm._do_training(indices)
            if smoke:
                if algorithm.loss["virtual_generated_tasks"] <= 0:
                    raise ValueError("Real smoke generated no virtual tasks")
                for key, enabled in (("R_encoder_gradient_norm", algorithm.mask_r), ("V_encoder_gradient_norm", algorithm.mask_v),
                                     ("P_critic_gradient_norm", algorithm.mask_p), ("P_actor_Q_gradient_norm", algorithm.mask_p),
                                     ("P_actor_BC_gradient_norm", algorithm.mask_p)):
                    if (algorithm.loss[key] > 0) != bool(enabled):
                        raise ValueError("Real smoke gradient-path failure: " + key)
                if algorithm.loss["base_encoder_gradient_norm"] <= 0:
                    raise ValueError("Base reconstruction no longer trains encoder")
            algorithm._n_train_steps_total += 1
            append_jsonl(run / "training_steps.jsonl", {"iteration": iteration, "main_update": algorithm._n_train_steps_total,
                                                       "update_counts": algorithm.update_counts.copy(), **algorithm.loss})
        algorithm.training_mode(False)
        algorithm.runtime_totals["training_seconds"] = algorithm.runtime_totals.get("training_seconds", 0.0) + time.monotonic() - iteration_start
        for key, value in decoder_before.items():
            if not torch.equal(value, algorithm.context_decoder.state_dict()[key].detach().cpu()):
                raise ValueError("Frozen decoder parameters changed")
        if any(parameter.grad is not None for parameter in algorithm.context_decoder.parameters()):
            raise ValueError("Frozen decoder received parameter gradients")
        elapsed = time.monotonic() - wall_start
        row = {"iteration": iteration, "main_updates": algorithm._n_train_steps_total, "update_counts": algorithm.update_counts.copy(),
               "real_batch_ids_sha256": algorithm.real_batch_digest.hexdigest(), "seconds_this_process": elapsed,
               "peak_cuda_bytes": 0 if args.cpu else torch.cuda.max_memory_allocated(), **algorithm.loss}
        append_jsonl(run / "progress.jsonl", row)
        if smoke or iteration in checkpoint_iterations:
            checkpoint = run / ("checkpoints/itr_%d.pt" % iteration)
            save_archive(checkpoint, training_state(algorithm, identity, iteration + 1, include_buffers=False))
            evaluation_start = time.monotonic()
            with isolated_rng("component-evaluation", args.env, args.seed, iteration):
                main_evaluation(algorithm, resolved, output, args.seed, iteration, smoke=smoke)
            algorithm.runtime_totals["evaluation_seconds"] = algorithm.runtime_totals.get("evaluation_seconds", 0.0) + time.monotonic() - evaluation_start
        # Alternate two complete states, then atomically advance the pointer.
        # A crash while publishing the next state leaves the prior one usable.
        resume_path = run / ("resume/slot%d.pt" % (iteration % 2))
        atomic_torch_save(resume_path, training_state(algorithm, identity, iteration + 1))
        write_json(run / "last_checkpoint.json", {"path": str(resume_path.relative_to(run)).replace("\\", "/"), "sha256": sha256(resume_path), "next_iteration": iteration + 1})
        print(json.dumps({"arm": args.arm, "seed": args.seed, "environment": args.env, "iteration": iteration,
                          "main_updates": algorithm._n_train_steps_total, "qf_loss": algorithm.loss["qf_loss"]}), flush=True)
    expected_updates = resolved["algo_params"]["num_iterations"] * resolved["algo_params"]["num_train_steps_per_itr"]
    if algorithm._n_train_steps_total != expected_updates:
        raise ValueError("Main update budget mismatch")
    multiplier = algorithm.embedding_batch_size // algorithm.embedding_mini_batch_size
    if algorithm.update_counts["actor"] != expected_updates * multiplier or algorithm.update_counts["critic"] != expected_updates * multiplier:
        raise ValueError("Optimizer update budget mismatch")
    history = run / "progress.jsonl"
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    columns = ["iteration", "main_updates", "seconds_this_process", "peak_cuda_bytes", "recon_loss", "recovery_real_raw", "recovery_virtual_raw", "recovery_real_weighted", "recovery_virtual_weighted", "qf_loss", "policy_total_loss", "virtual_transition_scheduled_weight", "virtual_transition_effective_weight"]
    with (run / "progress.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    final_iteration = resolved["algo_params"]["num_iterations"] - 1
    final_checkpoint = run / ("checkpoints/itr_%d.pt" % final_iteration)
    scientific_paths = [run / "progress.csv", run / "progress.jsonl", run / "training_steps.jsonl"]
    for iteration in ([0] if smoke else protocol["checkpoints"]):
        scientific_paths.extend([run / ("checkpoints/itr_%d.pt" % iteration), output / ("main_itr_%d.json" % iteration)])
    write_json(run / "completion.json", {"identity": identity, "complete": True, "kind": "smoke_only" if smoke else "formal_training",
                                         "created_at": now(), "final_iteration": final_iteration, "main_updates": expected_updates,
                                         "update_counts": algorithm.update_counts, "initial_sha256": init_manifest["sha256"],
                                         "final_checkpoint_sha256": sha256(final_checkpoint), "wall_seconds_this_process": time.monotonic() - wall_start,
                                         "runtime_totals_committed_iterations": algorithm.runtime_totals,
                                         "virtual_usage": algorithm.usage_totals,
                                         "outputs": output_hashes(run, scientific_paths),
                                         "peak_cuda_bytes": 0 if args.cpu else torch.cuda.max_memory_allocated(),
                                         "resume_boundary": "iteration; buffers/optimizers/targets/RNG restored", "decoder_verified_unchanged": True}, exclusive=True)
    if not smoke:
        write_json(run / "main_completion.json", {"identity": identity, "complete": True, "created_at": now(),
                                                   "checkpoint_sha256": sha256(final_checkpoint),
                                                   "outputs": output_hashes(run, [output / "main_itr_499.json"]), "input_binding": None}, exclusive=True)
