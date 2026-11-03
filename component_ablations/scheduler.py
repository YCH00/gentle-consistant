"""Bounded subprocess queue; GPU visibility is set before any CUDA import."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import ARMS
from .common import REPO, now, read_json, sha256, source_identity, status, write_json


def select_jobs(root, phase, environments=None, arms=None, seeds=None):
    matrix = read_json(Path(root) / "run_matrix.json")["slots"]
    jobs = []
    for row in matrix:
        if environments and row["environment"] not in environments:
            continue
        if arms and row["arm"] not in arms:
            continue
        if seeds is not None and row["seed"] not in seeds:
            continue
        if phase == "smoke" and seeds is None and row["seed"] != 0:
            continue
        if phase == "init" and row["arm"] != "full":
            continue
        jobs.append(row)
    return jobs


def make_command(args, row, gpu):
    command = [args.python, "-u", "-m", "component_ablations", "worker", "--root", str(Path(args.root).resolve()),
               "--phase", args.phase, "--env", row["environment"], "--arm", row["arm"], "--seed", str(row["seed"]), "--gpu", str(gpu)]
    if args.resume:
        command.append("--resume")
    if args.cpu:
        command.append("--cpu")
    return command


def process_environment(gpu, environment, mujoco_root, cpu=False):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" if cpu else str(gpu)
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    version = "131" if environment == "walker-rand-params" else "150"
    engine = Path(mujoco_root).expanduser() / ("mjpro" + version)
    env["MUJOCO_PY_MJPRO_PATH"] = str(engine)
    # Keep inherited driver/search directories while dropping the other engine.
    old = env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(engine / "bin")] + [item for item in old if item and "mjpro131" not in item and "mjpro150" not in item])
    return env


def completion_path(root, row, phase):
    if phase == "init":
        return Path(root) / row["initial_state"].replace("initial.pt", "manifest.json")
    if phase == "smoke":
        return Path(root) / "smoke" / row["slot_id"] / "completion.json"
    name = "completion.json" if phase == "train" else phase + "_completion.json"
    return Path(root) / row["run_directory"] / name


def check_gates(root, phase):
    from .artifacts import verify_plan, verify_completion
    root = Path(root)
    verify_plan(root)
    audit = read_json(root / "resource_audit.json")
    if not audit["resources_ready"] or audit["resources_sha256"] != sha256(root / "resources.json"):
        raise ValueError("Stage A resources are not verified; see resource_audit.json")
    if not all(audit["dependencies"].values()):
        raise ValueError("Stage A training dependencies are missing; rerun audit with the training interpreter")
    source = source_identity()["source_digest"]
    if audit["source"]["source_digest"] != source:
        raise ValueError("Source changed after audit")
    checks = read_json(root / "smoke/unit_checks.json")
    if not checks["passed"] or checks["source_digest"] != source:
        raise ValueError("Current numerical/gradient/state checks have not passed")
    if phase in ("train", "main", "fixed", "probe"):
        matrix = read_json(root / "run_matrix.json")["slots"]
        for env in read_json(root / "protocol.json")["environments"]:
            for arm in ARMS:
                path = root / ("smoke/%s/%s/seed0/completion.json" % (env, arm))
                if not path.exists():
                    raise ValueError("Stage B real environment smoke missing: " + str(path))
                row = next(slot for slot in matrix if slot["environment"] == env and slot["arm"] == arm and slot["seed"] == 0)
                verify_completion(root, row, "smoke")


def launch(args):
    from .artifacts import verify_completion
    root = Path(args.root)
    jobs = select_jobs(root, args.phase, args.envs, args.arms, args.seeds)
    if not jobs:
        raise ValueError("Selection contains no jobs")
    if args.dry_run:
        preview = [{"slot_id": row["slot_id"], "physical_gpu": args.gpus[i % len(args.gpus)],
                    "command": make_command(args, row, args.gpus[i % len(args.gpus)]),
                    "engine": "131" if row["environment"] == "walker-rand-params" else "150"} for i, row in enumerate(jobs)]
        print(json.dumps({"dry_run": True, "phase": args.phase, "jobs": len(preview), "jobs_per_gpu": args.jobs_per_gpu,
                          "max_parallel": args.max_parallel or len(args.gpus) * args.jobs_per_gpu, "queue": preview}, indent=2))
        return 0
    check_gates(root, args.phase)
    lock = root / ".scheduler.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "created_at": now(), "phase": args.phase}))
    running, history, failures = [], [], []
    matrix = read_json(root / "run_matrix.json")
    slot_map = {row["slot_id"]: row for row in matrix["slots"]}
    max_parallel = args.max_parallel or len(args.gpus) * args.jobs_per_gpu
    pending = jobs.copy()
    logs = root / "job_logs" / args.phase
    logs.mkdir(parents=True, exist_ok=True)
    try:
        while pending or running:
            for gpu in args.gpus:
                used = sum(item["gpu"] == gpu for item in running)
                while pending and used < args.jobs_per_gpu and len(running) < max_parallel:
                    row = pending.pop(0)
                    if args.phase == "train" and row["arm"] not in ARMS:
                        core_ready = all(slot["status"] == "completed" for slot in matrix["slots"] if slot["arm"] in ARMS)
                        if not core_ready:
                            if running:
                                pending.insert(0, row)
                                break
                            event = {"slot_id": row["slot_id"], "phase": args.phase, "status": "blocked",
                                     "reason": "All 48 core slots must complete before supplementary training"}
                            failures.append(event)
                            history.append(event)
                            slot_map[row["slot_id"]].update(status="blocked", reason=event["reason"])
                            continue
                    marker = completion_path(root, row, args.phase)
                    if marker.exists():
                        report = verify_completion(root, row, args.phase)
                        if args.phase == "train":
                            slot_map[row["slot_id"]].update(status="completed", model_sha256=report["final_checkpoint_sha256"],
                                                           initial_sha256=report["initial_sha256"], reason=None)
                        history.append({"slot_id": row["slot_id"], "status": "skipped_existing_verified_marker", "phase": args.phase})
                        continue
                    if args.phase == "train" and row["arm"] not in ARMS:
                        smoke_row = next(slot for slot in matrix["slots"] if slot["environment"] == row["environment"] and slot["arm"] == row["arm"] and slot["seed"] == 0)
                        verify_completion(root, smoke_row, "smoke")
                    command = make_command(args, row, gpu)
                    log = logs / (row["slot_id"].replace("/", "_") + ".log")
                    handle = log.open("a", encoding="utf-8")
                    handle.write("\n" + now() + " " + json.dumps(command) + "\n")
                    handle.flush()
                    process = subprocess.Popen(command, cwd=str(REPO), env=process_environment(gpu, row["environment"], args.mujoco_root, args.cpu),
                                               stdout=handle, stderr=subprocess.STDOUT)
                    running.append({"process": process, "handle": handle, "gpu": gpu, "row": row,
                                    "command": command, "log": str(log), "started_at": now()})
                    used += 1
                    if args.phase == "train":
                        slot_map[row["slot_id"]].update(status="running", physical_gpu=gpu, reason=None)
                    print("Started", row["slot_id"], "on GPU", gpu, "PID", process.pid, flush=True)
            for item in running[:]:
                code = item["process"].poll()
                if code is None:
                    continue
                item["handle"].close()
                running.remove(item)
                row = item["row"]
                marker = completion_path(root, row, args.phase)
                passed = code == 0 and marker.exists()
                validation_error = None
                if passed:
                    try:
                        verify_completion(root, row, args.phase)
                    except (ValueError, OSError, KeyError) as error:
                        passed = False
                        validation_error = str(error)
                event = {key: item[key] for key in ("gpu", "command", "log", "started_at")}
                event.update(slot_id=row["slot_id"], phase=args.phase, exit_code=code, status="completed" if passed else "failed", finished_at=now())
                if validation_error:
                    event["validation_error"] = validation_error
                history.append(event)
                if not passed:
                    failures.append(event)
                if args.phase == "train":
                    current = slot_map[row["slot_id"]]
                    current.update(status="completed" if passed else "failed", reason=None if passed else "See " + str(item["log"]))
                    if passed:
                        completion = read_json(marker)
                        current.update(model_sha256=completion["final_checkpoint_sha256"], initial_sha256=completion["initial_sha256"],
                                       data_manifest_sha256=completion["identity"]["resources_digest"])
                print("Finished", row["slot_id"], event["status"], "exit", code, flush=True)
                write_json(root / "run_matrix.json", matrix)
                write_json(root / ("jobs_" + args.phase + ".json"), {"phase": args.phase, "updated_at": now(), "jobs": history})
                status(root, active_phase=args.phase, queue={"pending": len(pending), "running": len(running), "completed_or_skipped": sum(h["status"] in ("completed", "skipped_existing_verified_marker") for h in history), "failed": len(failures)})
            if pending or running:
                time.sleep(0.2)
    except BaseException:
        for item in running:
            item["process"].terminate()
            try:
                item["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                item["process"].kill()
            item["handle"].close()
            if args.phase == "train":
                slot_map[item["row"]["slot_id"]].update(status="failed", reason="Scheduler interrupted; exact --resume requires last_checkpoint.json and its complete state")
        write_json(root / "run_matrix.json", matrix)
        status(root, active_phase=args.phase, interrupted=True)
        raise
    finally:
        lock.unlink()
    state = "FAILED: see jobs_%s.json" % args.phase if failures else "SELECTED_JOBS_COMPLETED"
    if args.phase == "smoke":
        status(root, B=state + "; formal training gate checks all twelve core smoke markers")
    elif args.phase == "train":
        complete = sum(row["status"] == "completed" for row in matrix["slots"])
        status(root, C={"status": state, "completed_slots": complete, "total_slots": len(matrix["slots"])})
    elif args.phase in ("main", "fixed", "probe"):
        status(root, D={"phase": args.phase, "status": state})
    write_json(root / ("jobs_" + args.phase + ".json"), {"phase": args.phase, "updated_at": now(), "jobs": history})
    write_json(root / "run_matrix.json", matrix)
    return 1 if failures else 0
