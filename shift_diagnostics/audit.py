"""Read-only resource discovery for the current shift-diagnostics protocol (Python >= 3.7).

This does NOT load checkpoints, import environments, select models, or certify
stage A. Candidate filenames are evidence of existence, never of provenance.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys


PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8"))
PROTOCOL_ID = PROTOCOL["protocol_id"]
METHODS = PROTOCOL["methods"]
ENVIRONMENTS = PROTOCOL["environments"]
SKIP_DIRS = {
    ".git", ".aws", ".ssh", ".codex", ".agents", "__pycache__", ".pytest_cache",
    ".venv", "venv", "site-packages", "node_modules", "99_backup",
    "shared-artifact-tests", "test-runtime", "plot-runtime", "%SystemDrive%",
    "diagnostics_audits", "shift-diagnostics-v1", "shift-diagnostics-v2",
}
METADATA_NAMES = {
    "variant.json", "config.json", "resolved_config.json", "run_manifest.json",
    "task_manifest.json", "dataset_manifest.json", "shared_artifact_usage.json",
    "preparation_provenance.json", "pretrain_metadata.json", "models.json",
    "protocol.json", "observation_normalizer.npz", "obs_normalizer.npz",
}
ENTRY_NAMES = {
    "evaluate_fixed_context.py", "evaluate_fixed_context_suite.py",
    "train_gentle.py", "train_main.py", "run_shared.py", "run_shared_native_v2.py",
    "launch_shared_native_v2.py", "run_gentle.py", "run_gentle_pipeline.py",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    # Output directory is created exclusively; no previous audit is overwritten.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_command(argv, cwd=None):
    try:
        proc = subprocess.run(argv, cwd=str(cwd) if cwd else None,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, errors="replace", timeout=30)
        return {"returncode": proc.returncode, "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": str(exc)}


def git_info(root):
    # Per-command trust exception only; never edit global Git configuration.
    prefix = ["git", "-c", "safe.directory=" + root.as_posix(), "-C", str(root)]
    result = {"path": str(root)}
    for label, args in [("head", ["rev-parse", "HEAD"]),
                        ("worktree", ["status", "--short", "--untracked-files=normal"])]:
        result[label] = run_command(prefix + args)
    return result


def file_record(path, hash_file=True):
    record = {"path": str(path)}
    try:
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256(path) if hash_file else None
        record["status"] = "present_hashed" if hash_file else "present_unhashed"
    except OSError as exc:
        record.update(status="unreadable", error=str(exc))
    return record


def walk_files(root, max_depth, excluded, errors):
    def onerror(exc):
        errors.append(str(exc))
    for current, dirs, names in os.walk(str(root), followlinks=False, onerror=onerror):
        current = Path(current)
        depth = len(current.relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                         and not d.startswith("test_")
                         and not (current / d).is_symlink()
                         and (current / d).resolve() not in excluded)
        if depth >= max_depth:
            dirs[:] = []
        for name in sorted(names):
            path = current / name
            if not path.is_symlink():
                yield path


def classify(path):
    name = path.name.lower()
    if path.name in METADATA_NAMES:
        return "shared_or_run_metadata"
    if path.name in ENTRY_NAMES:
        return "entrypoint"
    if path.suffix == ".py" and any(p in {"shared_protocol", "focal_shared", "shared_native"} for p in path.parts):
        return "shared_source"
    if path.suffix.lower() in {".pt", ".pth", ".pkl", ".ckpt"}:
        match = re.search(r"(?:itr|iter|iteration|epoch|step)[_-]?(\d+)", name)
        if match and int(match.group(1)) not in (200, 499):
            return None
        return "checkpoint_candidate"
    return None


def blank_models():
    entries = []
    for env in ENVIRONMENTS:
        for method in METHODS:
            for seed in range(4):
                entries.append(model_slot(method, env, seed, 499, ["evaluation"] + (
                    ["collector"] if method == "GENTLE" and seed == 0 else [])))
        entries.append(model_slot("GENTLE", env, 0, 200, ["collector"]))
    return {"protocol": PROTOCOL_ID, "status": "unresolved",
            "note": "missing means no verified assignment, not proof that the server file is absent",
            "expected_evaluation_models": len(METHODS) * len(ENVIRONMENTS) * 4,
            "expected_unique_model_slots": len(entries),
            "models": entries}


def model_slot(method, env, seed, iteration, roles):
    return {"method": method, "environment": env, "training_seed": seed,
            "iteration": iteration, "roles": roles,
            "status": "pending_training" if PROTOCOL.get("resource_status", {}).get(method + "/" + env) == "training_in_progress" else "missing",
            "user_selected_run_directory": PROTOCOL.get("run_selection", {}).get(method),
            "checkpoint_files": [], "training_config": None, "source_root": None,
            "source_commit": None, "source_tree_sha256": None, "worktree_summary": None,
            "run_id": None, "same_run_and_iteration_evidence": None,
            "full_method_evidence": None, "latent_dim": None,
            "context_input_fields": None, "posterior": None, "actor_interface": None,
            "observation_normalizer": None, "task_manifest": None,
            "preprocessing_contract": None, "adapter_status": "not_verified",
            "smoke_status": "not_run"}


def source_snapshot(root):
    """Hash code/config/XML only; skip datasets, results, credentials and binaries."""
    records, errors = [], []
    excluded = {root / name for name in ("output", "result", "logs", "data", "dynamics", "figures")}
    for path in walk_files(root, 12, excluded, errors):
        if path.suffix.lower() in {".py", ".json", ".yaml", ".yml", ".xml", ".sh", ".bash"} or path.name in {"AGENTS.md", "requirements.txt"}:
            record = file_record(path)
            record["path"] = path.relative_to(root).as_posix()
            records.append(record)
    records.sort(key=lambda item: item["path"])
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"root": str(root), "scope": "code/config/XML only; depth <= 12; artifact directories excluded",
            "files": records, "source_tree_sha256": hashlib.sha256(payload).hexdigest(), "errors": errors}


def installed_versions():
    try:
        from importlib import metadata
    except ImportError:
        try:
            import importlib_metadata as metadata
        except ImportError:
            return {"status": "metadata_api_unavailable; no packages imported"}
    versions = {}
    for name in ["numpy", "torch", "gym", "mujoco-py", "scikit-learn"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path,
                        help="Current GENTLE checkout; existence is verified")
    parser.add_argument("--search-root", action="append", type=Path, default=[],
                        help="Repeat for actual parent/shared/model directories; read-only")
    parser.add_argument("--out", required=True, type=Path, help="NEW audit directory; must not exist")
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--hash-checkpoints", action="store_true",
                        help="Hash discovered checkpoint candidates (can take time)")
    args = parser.parse_args(argv)
    root, out = args.repo_root.expanduser().resolve(), args.out.expanduser().resolve()
    if not root.is_dir():
        parser.error("repo-root is not a directory: " + str(root))
    if args.max_depth < 0:
        parser.error("max-depth must be nonnegative")
    if out.exists():
        parser.error("refusing to overwrite existing output: " + str(out))
    protocol_path = Path(__file__).resolve().with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    requested_roots = sorted(set([root] + [p.expanduser().resolve() for p in args.search_root]), key=str)
    roots = list(requested_roots)
    # Drop nested search roots so large trees are not scanned repeatedly.
    roots = [p for p in roots if not any(q != p and q in p.parents for q in roots)]
    out.mkdir(parents=True, exist_ok=False)
    errors, candidates, repo_roots, seen = [], [], {root}, set()
    searches = []
    for search_root in roots:
        searches.append({"path": str(search_root), "status": "present" if search_root.is_dir() else "missing"})
        if not search_root.is_dir():
            continue
        for path in walk_files(search_root, args.max_depth, {out}, errors):
            if path in seen:
                continue
            seen.add(path)
            if (path.parent / ".git").exists():
                repo_roots.add(path.parent)
            kind = classify(path)
            if kind:
                record = file_record(path, kind != "checkpoint_candidate" or args.hash_checkpoints)
                record["kind"] = kind
                record["assignment"] = "unverified"
                candidates.append(record)
    print("Discovered {} candidates; recording source snapshots.".format(len(candidates)), flush=True)
    snapshots = [source_snapshot(p) for p in sorted(repo_roots, key=str)]
    write_json(out / "source_snapshots.json", snapshots)
    write_json(out / "protocol.json", protocol)
    write_json(out / "models.json", blank_models())
    write_json(out / "candidates.json", candidates)
    audit = {
        "protocol": PROTOCOL_ID, "stage": "A", "stage_complete": False,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host_platform": platform.platform(), "python": sys.version,
        "python_executable": sys.executable, "installed_package_metadata": installed_versions(),
        "search_roots": searches, "max_depth": args.max_depth, "scan_errors": errors,
        "requested_roots": [{"path": str(p), "status": "present" if p.is_dir() else "missing"}
                            for p in requested_roots],
        "excluded_directory_names": sorted(SKIP_DIRS), "follow_symlinks": False,
        "checkpoint_hashes_requested": args.hash_checkpoints,
        "candidate_counts": {kind: sum(c["kind"] == kind for c in candidates) for kind in
                             ["checkpoint_candidate", "shared_or_run_metadata", "entrypoint", "shared_source"]},
        "repositories": [git_info(p) for p in sorted(repo_roots, key=str)],
        "local_entrypoints": {name: ("present" if (root / name).is_file() else "missing")
                              for name in sorted(ENTRY_NAMES)},
        "shared_python_sources": sorted(str(p) for p in (root / "shared_protocol").glob("*.py")),
        "protocol_source": file_record(protocol_path),
        "handoff_document": file_record(root / protocol["specification"]),
        "expected_handoff_sha256": protocol["specification_sha256"],
        "audit_script": file_record(Path(__file__).resolve()),
        "blockers": [
            "{} evaluation slots and {} additional collector-200 slots have no verified assignment".format(len(METHODS) * len(ENVIRONMENTS) * 4, len(ENVIRONMENTS)),
            "Shared task manifests, normalizers and full offline pools require provenance/format verification",
            "Native adapters and original-GENTLE vs Ours identity require actual training source/config",
            "Real-environment shared-context smoke has not run; B/C/D remain pending",
        ],
        "checks_not_performed": ["checkpoint deserialization", "model strict loading", "normalizer semantics",
                                 "task restoration", "MuJoCo reset/step", "shared-context smoke"],
    }
    write_json(out / "resource_audit.json", audit)
    command = " ".join(shlex.quote(arg) for arg in [sys.executable, str(Path(__file__).resolve())] + (
        list(argv) if argv is not None else sys.argv[1:]))
    (out / "commands.sh").write_text("#!/usr/bin/env bash\n# Recorded invocation; use a NEW --out to rerun.\n" + command + "\n", encoding="utf-8")
    status = """# STATUS — {protocol_id}

- A: IN_PROGRESS — resource discovery finished; assignments/adapters/smoke pending.
- B / C / D: NOT_STARTED.
- Only user-selected complete methods; no component ablations; one shared raw context bank.

## Executed command
```sh
{command}
```

## Outputs and validation
- Output directory: `{out}`
- `resource_audit.json`: search coverage, missing roots, errors, versions and Git state.
- `candidates.json`: {count} candidates; names do not establish method/seed/iteration identity.
- `source_snapshots.json`: source/config/XML hashes; no checkpoint deserialization.
- `models.json`: {total_slots} unresolved slots ({eval_slots} evaluation models + {collector_slots} collector-200 models).
- `protocol.json`: proposed fixed protocol; resource hashes remain unresolved.
- True-environment smoke: NOT_RUN; stage A is NOT complete.

## Next step
Review candidates and actual run configurations; assign all model slots and shared resources,
verify matching encoder/policy provenance and normalizer/task hashes, then implement native
adapters and run each method/environment on the SAME raw smoke contexts. Do not start B yet.
Append every later command, output path, validation evidence and next step here.
""".format(command=command, out=out, count=len(candidates), protocol_id=PROTOCOL_ID,
           total_slots=len(METHODS) * len(ENVIRONMENTS) * 4 + len(ENVIRONMENTS),
           eval_slots=len(METHODS) * len(ENVIRONMENTS) * 4, collector_slots=len(ENVIRONMENTS))
    (out / "STATUS.md").write_text(status, encoding="utf-8")
    outputs = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out / "audit_completion.json", {"resource_discovery_finished": True,
                                               "stage_A_complete": False, "output_sha256": outputs})
    print("Audit written to {}. Stage A incomplete; review candidates before model loading.".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
