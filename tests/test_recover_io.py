"""Recovery regressions use small checkpoints under pytest temporary paths."""
import copy
import json
from pathlib import Path

import pytest
import torch

import recover_component_runs as recovery
from component_ablations.common import read_json, sha256, source_identity, write_json
from component_ablations.state import atomic_torch_save, load_training_state, training_state
from test_component_ablations import TinyTraining


def state_at(iteration, identity):
    model = TinyTraining((1, 1, 1))
    model.itr = iteration - 1
    model._n_train_steps_total = iteration
    return training_state(model, identity, iteration)


@pytest.fixture
def checkpoints(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    identity = {"source": "fixed", "config": "fixed"}
    config = {"algo_params": {"num_iterations": 10, "num_train_steps_per_itr": 1}}
    previous, current = run / "resume/slot0.pt", run / "resume/slot1.pt"
    atomic_torch_save(previous, state_at(1, identity))
    atomic_torch_save(current, state_at(2, identity))
    write_json(run / "last_checkpoint.json", {"path": "resume/slot1.pt", "sha256": sha256(current), "next_iteration": 2})
    return run, previous, current, identity, config


def test_verified_pointer_is_preferred_to_unpublished_newer_state(checkpoints):
    run, previous, current, identity, config = checkpoints
    atomic_torch_save(previous, state_at(3, identity))
    selected, errors = recovery.choose_checkpoint(run, identity, config)
    assert selected["path"] == str(current) and selected["next_iteration"] == 2
    assert errors == []


@pytest.mark.parametrize("damage", ["truncated_state", "changed_state", "malformed_pointer", "missing_pointer"])
def test_fallback_after_interrupted_io_preserves_existing_files(checkpoints, damage):
    run, previous, current, identity, config = checkpoints
    pointer = run / "last_checkpoint.json"
    if damage == "truncated_state":
        current.write_bytes(b"incomplete torch file")
    elif damage == "changed_state":
        state = state_at(2, identity)
        state["networks"]["policy"][next(iter(state["networks"]["policy"]))].add_(0.1)
        atomic_torch_save(current, state)
    elif damage == "malformed_pointer":
        pointer.write_bytes(b'{"path":')
    else:
        pointer.unlink()
    originals = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    selected, errors = recovery.choose_checkpoint(run, identity, config)
    assert selected["next_iteration"] == (1 if damage in ("truncated_state", "changed_state") else 2)
    assert selected["path"] == str(previous if damage in ("truncated_state", "changed_state") else current)
    for path, data in originals.items():
        assert path.read_bytes() == data
    if damage != "missing_pointer":
        assert errors


def test_both_bad_states_are_blocked_and_network_only_snapshot_is_rejected(checkpoints):
    run, previous, current, identity, config = checkpoints
    previous.write_bytes(b"bad file")
    state = state_at(2, identity)
    state["buffers"] = {}
    atomic_torch_save(current, state)
    write_json(run / "last_checkpoint.json", {"path": "resume/slot1.pt", "sha256": sha256(current), "next_iteration": 2})
    selected, errors = recovery.choose_checkpoint(run, identity, config)
    assert selected is None
    assert any("replay-buffer" in error for error in errors)
    assert (run / "last_checkpoint.json").exists()


def test_fallback_pointer_publication_keeps_original_pointer_backup(checkpoints, tmp_path):
    run, previous, current, identity, config = checkpoints
    current.write_bytes(b"failed write")
    original = (run / "last_checkpoint.json").read_bytes()
    selected, errors = recovery.choose_checkpoint(run, identity, config)
    row = {"run_directory": "run", "slot_id": "env/arm/seed0"}
    backup = tmp_path / "recovery"
    recovery.prepare_pointer(tmp_path, row, {"action": "resume", "checkpoint": selected}, backup)
    assert (backup / row["slot_id"] / "last_checkpoint.json").read_bytes() == original
    assert read_json(run / "last_checkpoint.json")["sha256"] == sha256(previous)


def test_resume_from_selected_state_matches_uninterrupted_next_update(tmp_path):
    torch.manual_seed(91)
    identity = {"source": "fixed"}
    model = TinyTraining((1, 1, 1))
    context = torch.rand(2, 4, 3)
    model._take_step([0, 1], context)
    model._n_train_steps_total = 1
    path = tmp_path / "resume/slot0.pt"
    atomic_torch_save(path, training_state(model, identity, 1))
    write_json(tmp_path / "last_checkpoint.json", {"path": "resume/slot0.pt", "sha256": sha256(path), "next_iteration": 1})
    model._take_step([0, 1], context)
    expected = copy.deepcopy(model.get_epoch_snapshot(1))
    config = {"algo_params": {"num_iterations": 10, "num_train_steps_per_itr": 1}}
    selected, errors = recovery.choose_checkpoint(tmp_path, identity, config)
    restored = TinyTraining((1, 1, 1))
    load_training_state(restored, torch.load(selected["path"], map_location="cpu", weights_only=False), identity)
    restored._take_step([0, 1], context)
    for name, network in expected.items():
        for key, tensor in network.items():
            assert torch.equal(tensor, restored.get_epoch_snapshot(1)[name][key])
    assert not errors


def test_controller_mixes_resume_fresh_completed_and_blocked_jobs(tmp_path, monkeypatch):
    from component_ablations import artifacts, planning, reset, scheduler
    rows = [{"slot_id": "env/arm/seed%d" % seed, "environment": "cheetah-vel", "arm": "full", "seed": seed,
             "run_directory": "run%d" % seed, "status": "planned"} for seed in range(4)]
    write_json(tmp_path / "resource_audit.json", {"source": {"source_digest": "fixed"}})
    write_json(tmp_path / "run_matrix.json", {"slots": rows})
    monkeypatch.setattr(artifacts, "verify_plan", lambda root: None)
    monkeypatch.setattr(reset, "ensure_stopped", lambda root: None)
    monkeypatch.setattr(recovery, "source_identity", lambda: {"source_digest": "fixed"})
    monkeypatch.setattr(scheduler, "check_gates", lambda root, phase: None)
    monkeypatch.setattr(scheduler, "select_jobs", lambda *args: rows)
    monkeypatch.setattr(planning, "audit", lambda args: 0)
    actions = ["resume", "fresh", "completed", "blocked"]
    monkeypatch.setattr(recovery, "inspect_slot", lambda root, row: {"action": actions[row["seed"]],
                        "checkpoint": {"next_iteration": 3}, "reason": "failed checkpoint"})
    monkeypatch.setattr(recovery, "prepare_pointer", lambda *args: None)
    def launch(args):
        selected = scheduler.select_jobs(args.root, "train")
        assert [row["seed"] for row in selected] == [0, 1, 2]
        for row in selected:
            command = scheduler.make_command(args, row, "1")
            assert ("--resume" in command) == (row["seed"] == 0)
        return 0
    monkeypatch.setattr(scheduler, "launch", launch)
    args = recovery.parser().parse_args(["--root", str(tmp_path), "--group-seeds"])
    assert recovery.run(args) == 2
    assert read_json(tmp_path / "run_matrix.json")["slots"][3]["status"] == "blocked"
    assert len(list((tmp_path / "recovery").glob("*/plan.json"))) == 1


def test_recovery_tools_are_outside_frozen_training_source_fingerprint():
    files = source_identity()["source_files"]
    assert "recover_component_runs.py" not in files
    assert "tests/test_recover_io.py" not in files
