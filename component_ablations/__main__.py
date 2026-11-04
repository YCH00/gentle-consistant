import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

from . import ARMS
from .common import REPO, now, read_json, sha256, source_identity, status, verify_file, write_json


def gpus(value):
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("Use distinct comma-separated GPU IDs, e.g. 0,1,2,3")
    return result


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def config_differences(left, right, prefix=""):
    differences = []
    for key in set(left) | set(right):
        name = prefix + key
        if key not in left or key not in right:
            differences.append(name)
        elif isinstance(left[key], dict) and isinstance(right[key], dict):
            differences.extend(config_differences(left[key], right[key], name + "."))
        elif left[key] != right[key]:
            differences.append(name)
    return sorted(differences)


def check(args):
    root = Path(args.root)
    protocol = read_json(root / "protocol.json")
    differences = []
    for env in protocol["environments"]:
        full = read_json(root / "configs" / env / "full.json")
        for record in protocol["arms"]:
            arm = record["arm"]
            other = read_json(root / "configs" / env / (arm + ".json"))
            actual = config_differences(full, other)
            allowed = {"exp_name", "component_ablation.R", "component_ablation.V", "component_ablation.P"}
            if arm == "full_global":
                allowed.add("algo_params.virtual_task_generation_mode")
                allowed.update(("algo_params.M", "algo_params.beta"))
            if arm == "full_constant":
                allowed.add("algo_params.virtual_transition_weight_schedule")
            if set(actual) - allowed:
                raise ValueError("Non-whitelisted arm config difference: " + str(actual))
            if other["algo_params"]["n_vt"] <= 0 or not other["algo_params"]["virtual_transition_train_policy_bc"]:
                raise ValueError("Virtual generation/BC must be retained; only P masks contributions")
            differences.append({"environment": env, "arm": arm, "differences": actual})
    output = root / "smoke"
    output.mkdir(exist_ok=True)
    xml = output / "unit-tests.xml"
    command = [args.python, "-m", "pytest", "tests/test_component_ablations.py", "tests/test_component_orchestration.py", "-q", "-p", "no:cacheprovider", "--junitxml", str(xml.resolve())]
    process = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True, errors="replace")
    (output / "unit-checks.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    report = {"passed": process.returncode == 0, "exit_code": process.returncode, "created_at": now(),
              "source_digest": source_identity()["source_digest"], "command": command,
              "config_diff_whitelist": differences, "real_environment_smoke": "not covered by unit tests"}
    write_json(output / "unit_checks.json", report)
    status(root, B="NUMERICAL_CHECKS_PASSED; real MuJoCo smoke pending" if report["passed"] else "NUMERICAL_CHECKS_FAILED; see smoke/unit-checks.log",
           validation=report)
    print(process.stdout)
    if process.returncode:
        print(process.stderr, file=sys.stderr)
    return process.returncode


def bind(args):
    from .evaluation import validate_bank
    from .probe import validate_probe_manifest
    root = Path(args.root)
    resources = read_json(root / "resources.json")
    entry = resources[args.env]
    changes = {}
    for key, path, validator in (("diagnostic_bank", args.bank, validate_bank), ("probe_manifest", args.probe, validate_probe_manifest)):
        if path is not None:
            path = Path(path).resolve()
            manifest = read_json(path)
            validator(manifest)
            verify_file(manifest["arrays"], path.parent)
            if manifest.get("environment") != args.env or manifest.get("task_manifest_sha256") != entry["historical_task_manifest"]["sha256"]:
                raise ValueError("Binding environment/task-manifest mismatch")
            entry[key] = {"path": str(path), "sha256": sha256(path)}
            changes[key] = entry[key]
    if not changes:
        raise ValueError("Supply --bank and/or --probe")
    write_json(root / "resources.json", resources)
    status(root, diagnostic_bindings=changes, next_steps="Rerun audit after binding; training identities exclude diagnostic-only resources")


def parser():
    parser = argparse.ArgumentParser(description="R/V/P component ablations; 48 core or 64 total runs; separate from shift diagnostics.")
    sub = parser.add_subparsers(dest="command", required=True)
    def command(name, help_text):
        result = sub.add_parser(name, help=help_text)
        result.add_argument("--root", default="component-ablations-v1", help="Independent immutable experiment project")
        return result
    plan = command("plan", "Stage A: freeze resolved baseline, masks, loss contract and run matrix")
    from .planning import EVIDENCE
    plan.add_argument("--evidence", type=Path, default=EVIDENCE)
    plan.add_argument("--start-id", type=int, default=40)
    plan.add_argument("--name", default="component", help="Experiment names: 40_<name>_backbone, ..., 45_<name>_full")
    plan.add_argument("--suite", choices=("core", "all"), default="all", help="all adds Global and constant-weight full variants (64 slots)")
    plan.add_argument("--gpus", type=gpus, default=["0"])
    plan.add_argument("--jobs-per-gpu", type=positive, default=1)
    plan.add_argument("--data-root")
    plan.add_argument("--weights-root")
    plan.add_argument("--dynamics-root")
    audit_parser = command("audit", "Stage A: verify file hashes and dependencies; missing resources fail")
    audit_parser.add_argument("--seal-pretrained", action="store_true", help="Pin existing pretrained files' hashes once; never replace mismatched hashes")
    checks = command("check", "Stage B: numerical, gradient, resume and queue tests + config diff whitelist")
    checks.add_argument("--python", default=sys.executable)
    launch = command("launch", "Bounded GPU queue for init/smoke/train/main/fixed/probe")
    launch.add_argument("--phase", choices=("init", "smoke", "train", "main", "fixed", "probe"), required=True)
    launch.add_argument("--gpus", type=gpus, default=None)
    launch.add_argument("--jobs-per-gpu", type=positive, default=None)
    launch.add_argument("--max-parallel", type=positive)
    launch.add_argument("--group-seeds", action="store_true", help="Keep all selected seeds of one environment/arm on one GPU; reserve it until the whole group finishes")
    launch.add_argument("--envs", nargs="+", choices=("cheetah-vel", "walker-rand-params"))
    launch.add_argument("--arms", nargs="+", choices=(*ARMS, "full_global", "full_constant"))
    launch.add_argument("--seeds", nargs="+", type=int, choices=(0, 1, 2, 3))
    launch.add_argument("--python", default=sys.executable)
    launch.add_argument("--mujoco-root", default=str(Path.home() / ".mujoco"))
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--resume", action="store_true", help="Exact epoch-boundary restart from last_checkpoint.json and its verified complete state; network-only resume rejected")
    launch.add_argument("--cpu", action="store_true")
    worker = command("worker", "Internal one-slot worker; prefer launch for GPU binding and stage gates")
    worker.add_argument("--phase", choices=("init", "smoke", "train", "main", "fixed", "probe"), required=True)
    worker.add_argument("--env", choices=("cheetah-vel", "walker-rand-params"), required=True)
    worker.add_argument("--arm", choices=(*ARMS, "full_global", "full_constant"), required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--gpu", default="0")
    worker.add_argument("--resume", action="store_true")
    worker.add_argument("--cpu", action="store_true")
    command("summarize", "Stages D/E: actual four-seed results, paired differences and missing-resource status")
    command("plot", "Stage E: verified four-seed return bars/curves as PDF/SVG/PNG")
    reset = command("reset-runs", "Delete generated outputs/checkpoints and reset all slots for a fresh run; preserve frozen configs and resources")
    reset.add_argument("--dry-run", action="store_true", help="List output paths without deleting anything")
    binding = command("bind-diagnostics", "Bind formal frozen bank/probe manifests; reject smoke and task mismatches")
    binding.add_argument("--env", choices=("cheetah-vel", "walker-rand-params"), required=True)
    binding.add_argument("--bank", type=Path)
    binding.add_argument("--probe", type=Path)
    return parser


def main():
    args = parser().parse_args()
    try:
        if args.command == "plan":
            if args.start_id < 0 or not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
                raise ValueError("Use a nonnegative ID and an experiment name containing letters/numbers/_/-")
            from .planning import create_plan
            result = create_plan(args)
        elif args.command == "audit":
            from .planning import audit
            result = audit(args)
        elif args.command == "check":
            result = check(args)
        elif args.command == "launch":
            from .scheduler import launch
            protocol = read_json(Path(args.root) / "protocol.json")
            args.gpus = args.gpus or protocol["default_gpus"]
            args.jobs_per_gpu = args.jobs_per_gpu or protocol["jobs_per_gpu"]
            result = launch(args)
        elif args.command == "worker":
            os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.cpu else str(args.gpu)
            from .worker import run_worker
            # Defense in depth: direct workers still pass the same gates.
            from .scheduler import check_gates
            check_gates(args.root, args.phase)
            result = run_worker(args)
        elif args.command == "bind-diagnostics":
            result = bind(args)
        elif args.command == "plot":
            from .plotting import plot
            result = plot(args)
        elif args.command == "reset-runs":
            from .reset import reset_runs
            result = reset_runs(args)
        else:
            from .reporting import summarize
            result = summarize(args)
        return result or 0
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        if args.command == "worker":
            failures = Path(args.root) / "failures"
            failures.mkdir(parents=True, exist_ok=True)
            write_json(failures / ("%s_%s_%s_seed%d.json" % (args.phase, args.env, args.arm, args.seed)),
                       {"created_at": now(), "error": str(error), "traceback": traceback.format_exc()})
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
