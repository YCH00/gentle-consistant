"""Verify model identities before skipping jobs or combining result records."""
from pathlib import Path
from .common import digest_json, read_json, sha256, source_identity, verify_file


def verify_plan(root):
    root = Path(root)
    protocol = read_json(root / "protocol.json")
    if digest_json(read_json(root / "loss_contract.json")) != protocol["loss_contract_sha256"]:
        raise ValueError("Frozen loss contract changed")
    for env, expected in protocol["reference_config_sha256"].items():
        if sha256(root / "reference_configs" / (env + ".json")) != expected:
            raise ValueError("Frozen baseline resolved config changed: " + env)
    for row in read_json(root / "run_matrix.json")["slots"]:
        if sha256(root / row["config"]) != row["config_sha256"]:
            raise ValueError("Frozen arm config changed: " + row["config"])


def verify_completion(root, row, phase):
    from .scheduler import completion_path
    from .worker import initialization_identity
    root = Path(root)
    marker = completion_path(root, row, phase)
    report = read_json(marker)
    resources = read_json(root / "resources.json")[row["environment"]]
    expected = initialization_identity(root, row["environment"], row["seed"], resources)
    identity = report.get("identity", {})
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Completion source/contract/resource/seed identity mismatch: " + str(marker))
    initial = root / row["initial_state"]
    if phase == "init":
        if report["sha256"] != sha256(initial):
            raise ValueError("Initialization artifact hash mismatch")
        return report
    if report.get("complete") is not True or identity.get("config_sha256") != row["config_sha256"]:
        raise ValueError("Invalid completion/config marker: " + str(marker))
    if identity.get("initial_sha256") != sha256(initial):
        raise ValueError("Completion initialization hash mismatch")
    run = marker.parent
    if phase in ("train", "smoke"):
        expected_iteration = 0 if phase == "smoke" else 499
        if report["final_iteration"] != expected_iteration:
            raise ValueError("Incorrect final checkpoint iteration")
        expected_kind = "smoke_only" if phase == "smoke" else "formal_training"
        if report["kind"] != expected_kind:
            raise ValueError("Smoke/formal loss-family boundary violation")
        checkpoint_hash = report["final_checkpoint_sha256"]
    else:
        expected_iteration = 499
        checkpoint_hash = report["checkpoint_sha256"]
    if sha256(run / ("checkpoints/itr_%d.pt" % expected_iteration)) != checkpoint_hash:
        raise ValueError("Checkpoint hash mismatch: " + str(run))
    if not report.get("outputs"):
        raise ValueError("Completion is missing result artifact hashes")
    for artifact in report["outputs"]:
        verify_file(artifact, run)
    if phase in ("fixed", "probe"):
        key = "diagnostic_bank" if phase == "fixed" else "probe_manifest"
        if report.get("input_binding") != resources[key]:
            raise ValueError("Diagnostic input binding changed after evaluation")
        verify_file(resources[key], root)
    return report
