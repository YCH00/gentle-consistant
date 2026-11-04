import copy
import importlib.util
import shutil
from pathlib import Path

from . import ARMS, PAIRS, PROTOCOL
from .common import REPO, digest_json, now, read_json, sha256, source_identity, status, write_json

ENVIRONMENTS = ("cheetah-vel", "walker-rand-params")
EVIDENCE = REPO / "shift-diagnostics-v3/smoke/A-20261103T011150Z-15243"
if not EVIDENCE.exists():
    EVIDENCE = REPO / "component-ablations-v1/planning_evidence"


def create_plan(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists():
        raise FileExistsError("Plan already exists; use a new --root to change an immutable plan")
    evidence_files = ["models.json", "stage_a_plan.json"]
    evidence_files += ["banks/%s/%s" % (env, name) for env in ENVIRONMENTS
                       for name in ("legacy_training_normalizer.npz", "normalizer_provenance.json", "offline_pool_verification.json", "manifest.json")]
    absent = [relative for relative in evidence_files if not (args.evidence / relative).is_file()]
    if absent:
        raise FileNotFoundError("Incomplete planning evidence in %s: %s" % (args.evidence, ", ".join(absent)))
    models = read_json(args.evidence / "models.json")
    stage_a = read_json(args.evidence / "stage_a_plan.json")
    for relative in evidence_files:
        destination = root / "planning_evidence" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.evidence / relative, destination)
    selected = [m for m in models["models"] if m["method"] == "Ours-local"]
    configs, origins, resources, global_parameters = {}, {}, {}, {}
    for env in ENVIRONMENTS:
        candidates = [m for m in selected if m["environment"] == env]
        model = next(m for m in candidates if m["training_seed"] == 0)
        if {m["training_seed"] for m in candidates} != {0, 1, 2, 3}:
            raise ValueError("Historical Local evidence does not contain all four seeds: " + env)
        if any(m["configuration"]["algo_params"] != model["configuration"]["algo_params"] for m in candidates):
            raise ValueError("Historical Local seeds have different scientific settings: " + env)
        global_model = next(m for m in models["models"] if m["method"] == "Ours-global" and m["environment"] == env and m["training_seed"] == 0)
        global_parameters[env] = {key: global_model["configuration"]["algo_params"][key] for key in ("M", "beta")}
        if global_parameters[env]["M"] < 2:
            raise ValueError("Global comparison must mix at least two tasks")
        configs[env] = copy.deepcopy(model["configuration"])
        config = configs[env]
        # These are recorded RESOLVED full configs, not current defaults.
        config["algo_params"].update(virtual_transition_use_recon_weight=False,
                                     virtual_transition_use_cycle_weight=False,
                                     virtual_transition_log_nearest_distance=False,
                                     virtual_transition_train_policy_q=True,
                                     virtual_transition_train_policy_bc=True)
        config["seed_list"] = [0, 1, 2, 3]
        if args.data_root:
            config["algo_params"]["data_dir"] = str(Path(args.data_root) / env).replace("\\", "/")
        origins[env] = {"selected_run": model["run_directory"], "resolved_config_record": model["config_file"],
                        "evidence_sha256": sha256(args.evidence / "models.json"),
                        "seed_records": [{"seed": m["training_seed"], "config": m["config_file"],
                                          "checkpoint_records": m["checkpoint_files"]} for m in candidates],
                        "historical_source_version": "not proven by checkpoint hashes; source must be audited separately",
                        "four_seed_scientific_settings_match": True,
                        "global_generator_source": global_model["config_file"],
                        "reuse_historical_full": False}
        shared = stage_a["shared"][env]
        reference = root / "resources" / env
        reference.mkdir(parents=True)
        write_json(reference / "task_manifest.json", shared["task_payload"], exclusive=True)
        normalizer = args.evidence / "banks" / env / "legacy_training_normalizer.npz"
        shutil.copyfile(normalizer, reference / "normalizer.npz")
        native = read_json(args.evidence / "banks" / env / "normalizer_provenance.json")
        write_json(reference / "normalizer_provenance.json", native, exclusive=True)
        all_files = read_json(args.evidence / "banks" / env / "offline_pool_verification.json")["files"]
        by_key = {(item["task_id"], Path(item["path"]).name): item for item in all_files}
        ordered_training = [(int(Path(item["path"]).parent.name.replace("goal_idx", "")), Path(item["path"]).name) for item in native["native_loader_order"]]
        # Preserve the historical training-file order for normalization/buffer rows.
        keys = ordered_training + [key for key in by_key if key[0] in shared["task_payload"]["eval_tasks"]]
        files = []
        for key in keys:
            item = copy.deepcopy(by_key[key])
            item["path"] = str(Path(config["algo_params"]["data_dir"]) / ("goal_idx" + str(key[0])) / key[1]).replace("\\", "/")
            item["split"] = "train" if key[0] in shared["task_payload"]["train_tasks"] else "eval"
            files.append(item)
        server_repo = model["source_root"]
        seeds = {}
        for seed in (0, 1, 2, 3):
            weight_root = Path(args.weights_root or str(Path(server_repo) / "encoder_decoder")) / env / ("expert_seed" + str(seed))
            dynamics_root = Path(args.dynamics_root or str(Path(server_repo) / "dynamics")) / env / ("expert_seed" + str(seed))
            seeds[str(seed)] = {"encoder": {"path": str(weight_root / "context_encoder.pth").replace("\\", "/"), "sha256": None},
                                "decoder": {"path": str(weight_root / "context_decoder.pth").replace("\\", "/"), "sha256": None},
                                "dynamics": [{"path": str(dynamics_root / ("task%d_dynamics.pth" % i)).replace("\\", "/"), "sha256": None} for i in shared["task_payload"]["train_tasks"]]}
        resources[env] = {
            "tasks": {"path": "resources/%s/task_manifest.json" % env, "sha256": sha256(reference / "task_manifest.json")},
            "historical_task_manifest": shared["task_manifest.json"],
            "normalizer": {"path": "resources/%s/normalizer.npz" % env, "sha256": sha256(reference / "normalizer.npz")},
            "normalizer_origin": "hash-recorded reconstruction of historical single training-only RMS update; no context-bank reuse",
            "files": files, "pretrained": seeds,
            "backend_reference": read_json(args.evidence / "banks" / env / "manifest.json")["backend"],
            "diagnostic_bank": None, "probe_manifest": None,
        }
        write_json(root / "reference_configs" / (env + ".json"), config, exclusive=True)

    variants = list(ARMS)
    if args.suite == "all":
        variants += ["full_global", "full_constant"]
    variant_records = []
    matrix = []
    for offset, arm in enumerate(variants):
        masks = ARMS.get(arm, ARMS["full"])
        exp_name = "%d_%s_%s" % (args.start_id + offset, args.name, arm)
        variant_records.append({"arm": arm, "name": exp_name, "id": args.start_id + offset, "masks": dict(zip("RVP", masks))})
        for env in ENVIRONMENTS:
            config = copy.deepcopy(configs[env])
            config["exp_name"] = exp_name
            config["component_ablation"] = {"R": masks[0], "V": masks[1], "P": masks[2], "loss_family": "fixed-real-coefficient-with-virtual-bc", "virtual_bc": True}
            if arm == "full_global":
                config["algo_params"]["virtual_task_generation_mode"] = "global"
                config["algo_params"].update(global_parameters[env])
            elif arm == "full_constant":
                config["algo_params"]["virtual_transition_weight_schedule"] = "constant"
            config_path = "configs/%s/%s.json" % (env, arm)
            write_json(root / config_path, config, exclusive=True)
            for seed in (0, 1, 2, 3):
                slot_id = "%s/%s/seed%d" % (env, arm, seed)
                matrix.append({"slot_id": slot_id, "environment": env, "arm": arm, "seed": seed,
                               "experiment_name": exp_name, "config": config_path, "config_sha256": sha256(root / config_path),
                               "status": "planned", "reason": "Await resource verification and twelve real-environment smoke runs",
                               "run_directory": "runs/%s/%s/seed%d" % (env, exp_name, seed), "reuse": False,
                               "initial_state": "initial_states/%s/seed%d/initial.pt" % (env, seed),
                               "initial_sha256": None, "model_sha256": None, "data_manifest_sha256": None})
    contract = {
        "protocol_id": PROTOCOL, "loss_family": "fixed-real-coefficient-with-virtual-bc",
        "created_at": now(), "source_baselines": origins, "current_source": source_identity(),
        "historical_full_equivalent": False, "matched_full_retraining_required": True,
        "R_V": {"element_error": "(reencoded_z - stopgrad(target_z))**2; mean over latent elements and tasks",
                "context_length": "unchanged per-environment resolved embedding_mini_batch_size",
                "lambda_R": "consistency_loss_weight * actual_real_elements / actual_total_elements_before_masks",
                "lambda_V": "consistency_loss_weight * actual_virtual_elements / actual_total_elements_before_masks",
                "zero_virtual": "virtual contribution zero; coverage/skipped logged",
                "synthetic_context": "one combined construction under no_grad; re-encode with gradient",
                "base": "recon_loss_weight * mean((decoder(s,a,encoder(context))-target)**2); decoder frozen, encoder gradient preserved"},
        "critic": "mean_real((Q1-y)^2+(Q2-y)^2) + P*w(t)*mean_virtual(q*((Q1-y)^2+(Q2-y)^2))",
        "actor_Q": "-alpha*(mean_real(Q) + P*w(t)*mean_virtual(q*Q))",
        "alpha": "bc_weight / stopgrad(mean_real(abs(Q))).clamp(min=1e-6)",
        "actor_BC": "mean_real(mean_action((pi-a)^2)) + P*w(t)*mean_virtual(q*mean_action((pi-a)^2))",
        "virtual_bc": True,
        "virtual_bc_authority": "Explicit user reply 2026-11-03: retain virtual BC when P=1; supersedes handoff section 5.2",
        "quality": "q=1; cycle and reconstruction weighting disabled; no nearest-real distance logging",
        "generation_comparison": {"global_parameters": global_parameters,
                                  "reason_for_M_difference": "Local M=1 selects one nearest neighbor; Global M=2 mixes two tasks. M=1 Global would be degenerate. Parameters copied from historical Global evidence; all loss/budget/initialization settings remain matched."},
        "schedule": {env: {k: v for k, v in cfg["algo_params"].items() if "weight_schedule" in k or "weight_decay_" in k or k in ("virtual_transition_loss_weight", "virtual_transition_final_loss_weight")} for env, cfg in configs.items()},
        "gradient_paths": {"decoder_parameters": "frozen", "base_reconstruction_to_encoder": True,
                           "recovery_targets": "detached", "recovery_synthetic_context": "no_grad",
                           "virtual_RL_to_encoder": False, "R_V_reencoding_to_encoder": True},
        "update_schedule": {"main_updates": 200000, "actor_updates_per_main_update": 1,
                            "critic_optimizer_pairs_per_main_update": 1, "target_events_per_main_update": 1,
                            "target_networks_per_event": 3,
                            "policy_freq_note": "config says 2, but audited GENTLE._take_step updates actor and targets EVERY step; matched implementation preserves actual behavior"},
        "rng": "counter-keyed independent auxiliary streams; masks never skip generation; real stream restored; evaluation isolated",
        "resumption": "epoch-boundary complete state only; no native network-only resume",
        "differences_from_historical": ["independent real/virtual RL denominators", "real-only alpha normalization", "isolated auxiliary RNG", "strict resource loading and frozen task/normalizer inputs"],
    }
    protocol = {"protocol_id": PROTOCOL, "created_at": now(), "experiment_name": args.name,
                "start_id": args.start_id, "suite": args.suite, "arms": variant_records,
                "environments": list(ENVIRONMENTS), "training_seeds": [0, 1, 2, 3],
                "core_training_slots": 48, "total_training_slots": len(matrix),
                "baseline_generation": "local; locked before results", "loss_contract_sha256": digest_json(contract),
                "reference_config_sha256": {env: sha256(root / "reference_configs" / (env + ".json")) for env in ENVIRONMENTS},
                "default_gpus": args.gpus, "jobs_per_gpu": args.jobs_per_gpu,
                "checkpoints": [0, 200, 400, 499], "evaluation_iteration": 499,
                "paired_comparisons": PAIRS, "statistics": {"unit": "training_seed", "ddof": 1, "missing_seed": "incomplete; mean/std null"},
                "main_evaluation": "shared-main-v1, given-context and own-policy 200-transition self-collected pool; 600+ steps",
                "diagnostic_evaluation": {"conditions": ["offline", "policy499_sigma000", "policy499_sigma020"],
                                          "tasks": "all 10 seen + 10 unseen", "contexts_per_task": 20,
                                          "context_length": 200, "rollouts_per_context": 3,
                                          "collector": "original GENTLE; formal bank required; smoke rejected"},
                "separate_from": "shift-diagnostics complete-method experiments; no files there are modified"}
    write_json(root / "protocol.json", protocol, exclusive=True)
    write_json(root / "loss_contract.json", contract, exclusive=True)
    write_json(root / "resources.json", resources, exclusive=True)
    write_json(root / "run_matrix.json", {"protocol_id": PROTOCOL, "slots": matrix}, exclusive=True)
    commands = ["#!/usr/bin/env bash", "set -euo pipefail",
                '# Run from the repository, after editing paths in resources.json if necessary.',
                'ROOT="${1:-' + str(root).replace("\\", "/") + '}"',
                'GPUS="${2:-' + ",".join(args.gpus) + '}"',
                'PER_GPU="${3:-' + str(args.jobs_per_gpu) + '}"',
                'MODE="${4:-jobs}"',
                'GROUP_ARGS=()',
                'case "$MODE" in group-seeds) GROUP_ARGS=(--group-seeds);; jobs) ;; *) echo "Mode must be jobs or group-seeds" >&2; exit 2;; esac',
                'ABLATION_PYTHON="${ABLATION_PYTHON:-python}"',
                '"$ABLATION_PYTHON" -m component_ablations audit --root "$ROOT" --seal-pretrained',
                '"$ABLATION_PYTHON" -m component_ablations check --root "$ROOT"',
                '"$ABLATION_PYTHON" -m component_ablations launch --root "$ROOT" --phase init --gpus "$GPUS" --jobs-per-gpu "$PER_GPU" "${GROUP_ARGS[@]}"',
                '"$ABLATION_PYTHON" -m component_ablations launch --root "$ROOT" --phase smoke --gpus "$GPUS" --jobs-per-gpu "$PER_GPU"',
                '"$ABLATION_PYTHON" -m component_ablations launch --root "$ROOT" --phase train --gpus "$GPUS" --jobs-per-gpu "$PER_GPU" "${GROUP_ARGS[@]}"',
                '# Set ABLATION_DIAGNOSTICS=1 after binding both formal bank/probe manifests.',
                'if [ "${ABLATION_DIAGNOSTICS:-0}" = "1" ]; then',
                '  "$ABLATION_PYTHON" -m component_ablations launch --root "$ROOT" --phase fixed --gpus "$GPUS" --jobs-per-gpu "$PER_GPU" "${GROUP_ARGS[@]}"',
                '  "$ABLATION_PYTHON" -m component_ablations launch --root "$ROOT" --phase probe --gpus "$GPUS" --jobs-per-gpu "$PER_GPU" "${GROUP_ARGS[@]}"',
                'fi',
                '"$ABLATION_PYTHON" -m component_ablations plot --root "$ROOT"']
    with (root / "commands.sh").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(commands) + "\n")
    status(root, A="IN_PROGRESS: historical resolved Local configs and loss paths audited; resources pending actual verification",
           B="IMPLEMENTATION_PENDING_VALIDATION", C="NOT_STARTED", D="NOT_STARTED", E="NOT_STARTED",
           decisions={"virtual_bc": True, "retrain_matched_full": True, "numbering": variant_records},
           blockers=["No data/encoder_decoder/dynamics directories on this machine", "No formal fixed-context bank/probe manifest", "MuJoCo real smoke required"],
           commands=["python -m component_ablations plan --root " + str(root) + " --start-id " + str(args.start_id)])
    print("Planned", len(matrix), "slots in", root)


def audit(args):
    root = Path(args.root)
    resources = read_json(root / "resources.json")
    results, missing = [], []
    for env, entry in resources.items():
        records = [("tasks", entry["tasks"]), ("normalizer", entry["normalizer"])]
        records += [("data", item) for item in entry["files"]]
        for seed, pre in entry["pretrained"].items():
            records += [("pretrained/seed" + seed, pre["encoder"]), ("pretrained/seed" + seed, pre["decoder"])]
            records += [("dynamics/seed" + seed, item) for item in pre["dynamics"]]
        for kind, record in records:
            path = Path(record["path"])
            if not path.is_absolute():
                path = root / path
            result = {"environment": env, "kind": kind, "path": record["path"], "expected_sha256": record.get("sha256")}
            if not path.is_file():
                result["status"] = "missing"
            else:
                actual = sha256(path)
                if record.get("sha256") is None and args.seal_pretrained and kind.startswith(("pretrained/", "dynamics/")):
                    record["sha256"] = actual
                    result["expected_sha256"] = actual
                    result["sealed_now"] = True
                result["actual_sha256"] = actual
                result["status"] = "verified" if record.get("sha256") == actual else "hash_mismatch_or_unsealed"
            results.append(result)
            if result["status"] != "verified":
                missing.append(result)
    if args.seal_pretrained:
        write_json(root / "resources.json", resources)
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("torch", "numpy", "gym", "gtimer", "click", "matplotlib", "sklearn", "tensorboard")}
    report = {"protocol_id": PROTOCOL, "created_at": now(), "resource_count": len(results),
              "verified_count": len(results) - len(missing), "missing_or_mismatch_count": len(missing),
              "resources_ready": not missing, "dependencies": dependencies, "source": source_identity(),
              "resources_sha256": sha256(root / "resources.json"), "records": results,
              "formal_diagnostic_bank_ready": False, "probe_ready": False,
              "historical_full_reusable": False, "reason": "new matched RL denominator/alpha contract requires all six arms retrained"}
    write_json(root / "resource_audit.json", report)
    matrix = read_json(root / "run_matrix.json")
    for slot in matrix["slots"]:
        if slot["status"] not in ("planned", "blocked"):
            continue
        relevant = [item for item in missing if item["environment"] == slot["environment"] and
                    (item["kind"] in ("data", "tasks", "normalizer") or item["kind"].endswith("/seed" + str(slot["seed"])))]
        if relevant:
            slot.update(status="blocked", reason="Stage A: %d missing/unsealed/mismatched resources for this slot; see resource_audit.json" % len(relevant))
        elif not all(dependencies.values()):
            slot.update(status="blocked", reason="Training dependencies missing: " + ", ".join(name for name, present in dependencies.items() if not present))
        elif slot["status"] == "blocked" and str(slot.get("reason", "")).startswith(("Stage A:", "Training dependencies")):
            slot.update(status="planned", reason="Resources verified; awaiting required initialization/smoke")
    write_json(root / "run_matrix.json", matrix)
    status(root, A="RESOURCES_VERIFIED_PENDING_REAL_SMOKE" if not missing else "RESOURCE_BLOCKED; independent implementation/testing continues",
           resource_audit={k: report[k] for k in ("resource_count", "verified_count", "missing_or_mismatch_count", "dependencies", "resources_ready")},
           next_steps="audit --seal-pretrained on the training host, check, init, twelve real smoke jobs, then train")
    print("Resource audit:", report["verified_count"], "/", len(results), "verified; dependencies:", dependencies)
    return 0 if not missing and all(dependencies.values()) else 2
