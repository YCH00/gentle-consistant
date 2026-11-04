"""Explicitly clear one experiment's generated state for a fresh run."""
import os
from pathlib import Path
import shutil

from . import PROTOCOL
from .artifacts import verify_plan
from .common import now, read_json, status


OUTPUT_DIRECTORIES = ("runs", "initial_states", "smoke", "job_logs", "failures", "summaries", "figures")
OUTPUT_FILES = ("resource_audit.json", "stage_status.json", "STATUS.md", ".scheduler.lock")


def ensure_stopped(root):
    """Do not delete files while an old scheduler/worker can still write them."""
    root = Path(root).resolve()
    lock = root / ".scheduler.lock"
    if lock.exists():
        pid = int(read_json(lock)["pid"])
        if pid <= 0:
            raise ValueError("Invalid scheduler PID in " + str(lock))
        if os.name != "posix":
            raise ValueError("Scheduler lock exists; verify the old scheduler has stopped before removing the lock")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        except PermissionError:
            raise ValueError("Cannot verify scheduler stopped; PID %d exists" % pid)
        else:
            raise ValueError("Scheduler PID %d is still alive; stop it before resetting" % pid)
    proc = Path("/proc")
    if os.name == "posix" and proc.is_dir():
        for folder in proc.iterdir():
            if not folder.name.isdigit() or int(folder.name) == os.getpid():
                continue
            try:
                arguments = (folder / "cmdline").read_bytes().decode(errors="replace").split("\0")
                if "component_ablations" not in arguments or "worker" not in arguments:
                    continue
                argument = arguments[arguments.index("--root") + 1]
                base = (folder / "cwd").resolve()
                worker_root = (base / argument).resolve()
            except (OSError, ValueError, IndexError):
                continue
            if worker_root == root:
                raise ValueError("Worker PID %s still uses this project; stop it before resetting" % folder.name)


def reset_runs(args):
    root = Path(args.root).resolve()
    protocol = read_json(root / "protocol.json")
    if protocol.get("protocol_id") != PROTOCOL:
        raise ValueError("Refusing to reset an unrecognized experiment directory")
    verify_plan(root)
    matrix = read_json(root / "run_matrix.json")
    if matrix.get("protocol_id") != PROTOCOL:
        raise ValueError("Run matrix protocol mismatch")
    candidates = [root / name for name in OUTPUT_DIRECTORIES + OUTPUT_FILES]
    candidates.extend(root.glob("jobs_*.json"))
    targets = []
    for path in candidates:
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError("Refusing to delete linked or external output path: " + str(path))
        if path.exists():
            targets.append(path)
    ensure_stopped(root)
    if args.dry_run:
        print("Would reset %d slots in %s" % (len(matrix["slots"]), root))
        for path in targets:
            print("Would remove:", path)
        return 0
    # All deletion targets were resolved and verified within this named project.
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print("Removed:", path)
    for row in matrix["slots"]:
        row.update(status="planned", reason="Fresh restart: audit, checks, initialization and smoke required", reuse=False,
                   initial_sha256=None, model_sha256=None, data_manifest_sha256=None)
        row.pop("physical_gpu", None)
    from .common import write_json
    write_json(root / "run_matrix.json", matrix)
    status(root, A="NOT_STARTED: resource bindings preserved; rerun audit", B="NOT_STARTED", C="NOT_STARTED",
           D="NOT_STARTED", E="NOT_STARTED",
           reset={"at": now(), "slots": len(matrix["slots"]), "removed": [path.name for path in targets],
                  "preserved": ["protocol.json", "loss_contract.json", "configs/", "reference_configs/",
                                "resources.json", "resources/", "planning_evidence/", "commands.sh"],
                  "mode": "fresh start; no checkpoint or initialization reuse"})
    print("Reset complete. Frozen experiment settings and resource bindings preserved.")
    return 0
