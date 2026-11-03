"""Planning and audit utilities. No training dependencies are imported here."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(data)
    else:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seed_for(*parts):
    return int(digest_json(list(parts))[:8], 16)


def source_identity():
    files = [REPO / "train_gentle.py", REPO / "run_component_ablations.py", REPO / "configs/default.py"]
    files.extend((REPO / "tests").glob("test_component*.py"))
    for directory in ("component_ablations", "rlkit", "rand_param_envs"):
        files.extend((REPO / directory).rglob("*.py"))
    sources = {str(p.relative_to(REPO)).replace("\\", "/"): sha256(p) for p in sorted(files)}
    def git(*args):
        result = subprocess.run(["git", "-c", "safe.directory=" + REPO.as_posix(), *args], cwd=str(REPO), capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    return {"git_head": git("rev-parse", "HEAD"), "worktree_status": git("status", "--short"),
            "source_files": sources, "source_digest": digest_json(sources)}


def verify_file(record, root=None):
    path = Path(record["path"])
    if root is not None and not path.is_absolute():
        path = Path(root) / path
    if not path.is_file():
        raise FileNotFoundError(str(path))
    actual = sha256(path)
    if not record.get("sha256") or actual != record["sha256"]:
        raise ValueError("SHA256 mismatch or unsealed resource: " + str(path))
    return path


def status(root, **updates):
    root = Path(root)
    path = root / "stage_status.json"
    state = read_json(path) if path.exists() else {}
    state.update(updates)
    state["updated_at"] = now()
    write_json(path, state)
    lines = ["# component-ablations-v1 — STATUS", "", "Updated (UTC): " + state["updated_at"], ""]
    for key, value in state.items():
        if key != "updated_at":
            lines.extend(["## " + key, "", json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value, ""])
    (root / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")
