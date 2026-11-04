"""Resume interrupted component experiments with the existing training source."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
from types import SimpleNamespace
import zipfile

from component_ablations import ARMS
from component_ablations.common import digest_json, now, read_json, sha256, source_identity, write_json


def checked_path(run, relative):
    path = (run / relative).resolve()
    if run.resolve() not in path.parents:
        raise ValueError("Checkpoint path is outside its run directory")
    return path


def inspect_checkpoint(path, identity, config, expected_hash=None):
    import torch
    from component_ablations.state import BUFFERS, OPTIMIZERS

    before = path.stat()
    actual_hash = sha256(path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError("Checkpoint SHA256 differs from its committed pointer")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Checkpoint ZIP integrity check failed")
    elif expected_hash is None:
        raise ValueError("An unpinned fallback requires a complete ZIP checkpoint")
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    try:
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary")
        if state.get("format") != "component-ablations-v1-complete-state-v1" or state.get("identity") != identity:
            raise ValueError("Checkpoint format/source/config/initialization identity mismatch")
        for key in ("buffers", "optimizers", "networks", "counters", "schedule", "rng", "agent_buffers", "update_counts"):
            if not isinstance(state.get(key), dict):
                raise ValueError("Checkpoint is missing a state dictionary: " + key)
        if set(state.get("buffers", {})) != set(BUFFERS):
            raise ValueError("Checkpoint has no complete replay-buffer state")
        if set(state.get("optimizers", {})) != set(OPTIMIZERS):
            raise ValueError("Checkpoint optimizer state is incomplete")
        names = {"qf1", "qf2", "target_qf1", "target_qf2", "policy", "target_policy", "context_encoder", "context_decoder"}
        if set(state.get("networks", {})) != names:
            raise ValueError("Checkpoint network state is incomplete")
        for network in state["networks"].values():
            if not network or any(not torch.isfinite(tensor).all().item() for tensor in network.values()):
                raise ValueError("Checkpoint network contains missing or nonfinite parameters")
        iteration = state.get("next_iteration")
        parameters = config["algo_params"]
        if type(iteration) is not int or not 1 <= iteration <= parameters["num_iterations"]:
            raise ValueError("Checkpoint iteration is outside its training budget")
        if state["counters"]["_n_train_steps_total"] != iteration * parameters["num_train_steps_per_itr"]:
            raise ValueError("Checkpoint update counter disagrees with its iteration")
        if state["schedule"]["iteration"] != iteration - 1:
            raise ValueError("Checkpoint schedule disagrees with its iteration")
        for key in ("rng", "agent_buffers", "update_counts"):
            if not state.get(key):
                raise ValueError("Checkpoint is missing " + key)
    finally:
        del state
        gc.collect()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Checkpoint changed during verification")
    return {"path": str(path), "sha256": actual_hash, "next_iteration": iteration}


def choose_checkpoint(run, identity, config):
    errors = []
    pointer_path = run / "last_checkpoint.json"
    rejected = set()
    committed_iteration = None
    if pointer_path.exists():
        try:
            pointer = read_json(pointer_path)
            if not isinstance(pointer, dict):
                raise ValueError("Checkpoint pointer must be a dictionary")
            path = checked_path(run, pointer["path"])
            # A file that fails its committed hash must never be accepted as
            # an unpinned fallback in the same recovery attempt.
            rejected.add(path)
            if type(pointer.get("next_iteration")) is int:
                committed_iteration = pointer["next_iteration"]
            selected = inspect_checkpoint(path, identity, config, expected_hash=pointer["sha256"])
            if selected["next_iteration"] != pointer["next_iteration"]:
                raise ValueError("Pointer iteration disagrees with checkpoint")
            return selected, errors
        except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
            errors.append("last_checkpoint.json: " + str(error))
    candidates = []
    for name in ("resume/slot0.pt", "resume/slot1.pt"):
        path = checked_path(run, name)
        if path in rejected or not path.exists():
            continue
        try:
            candidate = inspect_checkpoint(path, identity, config)
            if committed_iteration is not None and candidate["next_iteration"] > committed_iteration:
                raise ValueError("Fallback is newer than the last published iteration")
            candidates.append(candidate)
        except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
            errors.append(name + ": " + str(error))
    if not candidates:
        return None, errors
    return max(candidates, key=lambda item: item["next_iteration"]), errors


def inspect_slot(root, row):
    from component_ablations.artifacts import verify_completion
    from component_ablations.worker import initialization_identity

    run = root / row["run_directory"]
    if (run / "completion.json").exists():
        verify_completion(root, row, "train")
        return {"action": "completed"}
    initial = verify_completion(root, row, "init")
    if not run.exists() or not any(run.iterdir()):
        return {"action": "fresh"}
    config = read_json(root / row["config"])
    resources = read_json(root / "resources.json")[row["environment"]]
    identity = {**initialization_identity(root, row["environment"], row["seed"], resources),
                "config_sha256": row["config_sha256"], "resolved_digest": digest_json(config),
                "initial_sha256": initial["sha256"], "kind": "formal"}
    if read_json(run / "provenance.json")["identity"] != identity:
        raise ValueError("Run provenance/source/config differs from the current experiment")
    if digest_json(read_json(run / "resolved_config.json")) != digest_json(config):
        raise ValueError("Resolved run configuration differs from the frozen configuration")
    checkpoint, errors = choose_checkpoint(run, identity, config)
    if checkpoint is None:
        return {"action": "blocked", "reason": "No verified complete checkpoint. Existing run files preserved.", "checkpoint_errors": errors}
    return {"action": "resume", "checkpoint": checkpoint, "checkpoint_errors": errors}


def prepare_pointer(root, row, decision, backup):
    if decision["action"] != "resume":
        return
    run = root / row["run_directory"]
    checkpoint = decision["checkpoint"]
    path = Path(checkpoint["path"])
    # Verify again immediately before publication, including a fallback whose
    # hash was computed during the initial recovery scan.
    if sha256(path) != checkpoint["sha256"]:
        raise ValueError("Checkpoint changed after the recovery scan")
    pointer = {"path": path.relative_to(run).as_posix(), "sha256": checkpoint["sha256"],
               "next_iteration": checkpoint["next_iteration"]}
    previous = run / "last_checkpoint.json"
    if previous.exists():
        if read_json_safely(previous) == pointer:
            return
        saved = backup / row["slot_id"] / "last_checkpoint.json"
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous, saved)
    write_json(previous, pointer)


def read_json_safely(path):
    try:
        return read_json(path)
    except (ValueError, OSError):
        return None


def run(args):
    from component_ablations import scheduler
    from component_ablations.artifacts import verify_plan
    from component_ablations.planning import audit
    from component_ablations.reset import ensure_stopped

    root = Path(args.root).resolve()
    verify_plan(root)
    ensure_stopped(root)
    previous_audit = read_json(root / "resource_audit.json")
    current_source = source_identity()
    if previous_audit["source"]["source_digest"] != current_source["source_digest"]:
        raise ValueError("Training source changed since the interrupted run. Restore that source before resuming.")
    scheduler.check_gates(root, "train")
    rows = scheduler.select_jobs(root, "train", args.envs, args.arms, args.seeds)
    if not rows:
        raise ValueError("Selection contains no training jobs")
    decisions = {}
    for row in rows:
        try:
            decision = inspect_slot(root, row)
        except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
            decision = {"action": "blocked", "reason": str(error)}
        decisions[row["slot_id"]] = decision
        detail = ""
        if decision["action"] == "resume":
            detail = " next_iteration=%d" % decision["checkpoint"]["next_iteration"]
        if decision["action"] == "blocked":
            detail = " " + decision["reason"]
        print(decision["action"].upper(), row["slot_id"], detail, flush=True)
        for error in decision.get("checkpoint_errors", []):
            print("  " + error, flush=True)
    counts = {name: sum(item["action"] == name for item in decisions.values())
              for name in ("completed", "resume", "fresh", "blocked")}
    print(json.dumps(counts), flush=True)
    if args.dry_run:
        return 0 if not counts["blocked"] else 2
    # Re-read and hash the actual data/weights before workers use the repaired
    # storage. This also refreshes the audit without changing training source.
    if audit(SimpleNamespace(root=str(root), seal_pretrained=False)):
        raise ValueError("Resource re-audit failed after storage recovery")
    backup = root / "recovery" / now().replace(":", "-")
    backup.mkdir(parents=True, exist_ok=False)
    write_json(backup / "plan.json", {"created_at": now(), "source_digest": current_source["source_digest"],
                                    "counts": counts, "slots": decisions})
    for row in rows:
        prepare_pointer(root, row, decisions[row["slot_id"]], backup)
    stale_lock = root / ".scheduler.lock"
    if stale_lock.exists():
        ensure_stopped(root)
        shutil.move(str(stale_lock), str(backup / "scheduler.lock"))
    matrix = read_json(root / "run_matrix.json")
    for row in matrix["slots"]:
        if decisions.get(row["slot_id"], {}).get("action") == "blocked":
            row.update(status="blocked", reason="Recovery: " + decisions[row["slot_id"]]["reason"])
    write_json(root / "run_matrix.json", matrix)
    selected = [row for row in rows if decisions[row["slot_id"]]["action"] != "blocked"]
    if not selected:
        return 2
    original_command, original_selection = scheduler.make_command, scheduler.select_jobs
    def command_for_slot(queue_args, row, gpu):
        worker_args = copy.copy(queue_args)
        worker_args.resume = decisions[row["slot_id"]]["action"] == "resume"
        return original_command(worker_args, row, gpu)
    try:
        scheduler.make_command = command_for_slot
        scheduler.select_jobs = lambda *unused_args, **unused_kwargs: selected
        result = scheduler.launch(args)
    finally:
        scheduler.make_command, scheduler.select_jobs = original_command, original_selection
    return result or (2 if counts["blocked"] else 0)


def parser():
    from component_ablations.__main__ import gpus, positive
    result = argparse.ArgumentParser(description="Verify interrupted runs and continue with per-slot resume decisions")
    result.add_argument("--root", default="component-ablations-v1")
    result.add_argument("--gpus", type=gpus, default=["0", "1", "2"])
    result.add_argument("--jobs-per-gpu", type=positive, default=4)
    result.add_argument("--max-parallel", type=positive)
    result.add_argument("--group-seeds", action="store_true")
    result.add_argument("--envs", nargs="+", choices=("cheetah-vel", "walker-rand-params"))
    result.add_argument("--arms", nargs="+", choices=(*ARMS, "full_global", "full_constant"))
    result.add_argument("--seeds", nargs="+", type=int, choices=(0, 1, 2, 3))
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--mujoco-root", default=str(Path.home() / ".mujoco"))
    result.add_argument("--dry-run", action="store_true")
    result.set_defaults(phase="train", resume=False, cpu=False)
    return result


if __name__ == "__main__":
    try:
        sys.exit(run(parser().parse_args()))
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
