"""Planner/scheduler/aggregation checks; all generated fixtures live in tmp."""
import copy
import os
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

from component_ablations import ARMS
from component_ablations.__main__ import config_differences
from component_ablations.common import read_json, write_json
from component_ablations.planning import EVIDENCE, create_plan, audit
from component_ablations.reporting import complete_statistics, aggregate_main, summarize
from component_ablations.scheduler import select_jobs, process_environment, make_command


@pytest.fixture
def project(tmp_path):
    args = SimpleNamespace(root=str(tmp_path / "component-project"), evidence=EVIDENCE, start_id=40,
                           name="exp44", suite="all", gpus=["2", "5"], jobs_per_gpu=1,
                           data_root=None, weights_root=None, dynamics_root=None)
    create_plan(args)
    return Path(args.root)


def test_plan_64_unique_slots_masks_and_frozen_source_config(project):
    assert b"\r" not in (project / "commands.sh").read_bytes()
    assert (project / "planning_evidence/models.json").is_file()
    protocol = read_json(project / "protocol.json")
    assert len(protocol["arms"]) == 8
    assert [row["id"] for row in protocol["arms"]] == list(range(40, 48))
    slots = read_json(project / "run_matrix.json")["slots"]
    assert len(slots) == len({row["slot_id"] for row in slots}) == 64
    for env in protocol["environments"]:
        full = read_json(project / "configs" / env / "full.json")
        global_config = read_json(project / "configs" / env / "full_global.json")
        assert global_config["algo_params"]["M"] == 2
        for arm, masks in ARMS.items():
            config = read_json(project / "configs" / env / (arm + ".json"))
            assert tuple(config["component_ablation"][key] for key in "RVP") == masks
            assert config["algo_params"]["n_vt"] == (20 if env == "cheetah-vel" else 10)
            assert config["algo_params"]["virtual_task_generation_mode"] == "local"
            assert config["algo_params"]["virtual_transition_train_policy_bc"] is True
            assert not config["algo_params"]["virtual_transition_use_recon_weight"]
            assert set(config_differences(full, config)) <= {"exp_name", "component_ablation.R", "component_ablation.V", "component_ablation.P"}
    with pytest.raises(FileExistsError):
        create_plan(SimpleNamespace(root=str(project)))


def test_queue_filters_and_binding_precede_torch_import(project, monkeypatch):
    assert len(select_jobs(project, "init")) == 8
    assert len(select_jobs(project, "smoke")) == 16
    assert len(select_jobs(project, "train", arms=list(ARMS))) == 48
    assert len(select_jobs(project, "train", environments=["walker-rand-params"], arms=["full"], seeds=[2])) == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join(["/old/mjpro150/bin", "/driver/lib"]))
    env = process_environment("5", "walker-rand-params", "/engines")
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert env["MUJOCO_PY_MJPRO_PATH"].replace("\\", "/").endswith("/engines/mjpro131")
    assert "mjpro150" not in env["LD_LIBRARY_PATH"] and "driver" in env["LD_LIBRARY_PATH"]
    args = SimpleNamespace(root=str(project), python="python", phase="train", resume=False, cpu=False)
    row = select_jobs(project, "train")[0]
    command = make_command(args, row, "5")
    assert command[command.index("--gpu") + 1] == "5"
    assert command[command.index("--seed") + 1] == "0"


def test_resource_audit_records_missing_instead_of_claiming_success(project):
    result = audit(SimpleNamespace(root=str(project), seal_pretrained=False))
    report = read_json(project / "resource_audit.json")
    assert result != 0 and report["resources_ready"] is False
    assert report["missing_or_mismatch_count"] > 0
    assert report["historical_full_reusable"] is False
    assert read_json(project / "loss_contract.json")["virtual_bc"] is True


def test_statistics_never_turn_missing_seeds_into_complete_result(project):
    partial = complete_statistics({"0": 3.0, "1": 5.0}, {"0": "completed", "1": "completed"})
    assert partial["n_seeds"] == 2 and partial["mean"] is None and partial["sample_std"] is None
    full = complete_statistics({str(i): float(i) for i in range(4)}, {str(i): "completed" for i in range(4)})
    assert full["complete"] and full["mean"] == 1.5
    assert full["sample_std"] == pytest.approx((5 / 3) ** 0.5)
    summarize(SimpleNamespace(root=str(project)))
    cells = read_json(project / "summaries/main_results.json")
    assert len(cells) == 32
    assert all(not cell["complete"] and cell["mean"] is None and cell["n_seeds"] == 0 for cell in cells)


def test_main_aggregation_weights_tasks_equally():
    records = []
    for task in range(10, 20):
        for episode in range(1 if task == 10 else 3):
            records.append({"protocol": "main_given_context", "split": "unseen", "task_id": task,
                            "repeat": 0, "return": float(task), "episode": episode})
    assert aggregate_main(records, "main_given_context") == 14.5
    with pytest.raises(ValueError):
        aggregate_main(records[:-3], "main_given_context")


def test_diagnostic_and_probe_manifests_reject_smoke_and_leakage():
    from component_ablations.evaluation import validate_bank
    from component_ablations.probe import validate_probe_manifest
    with pytest.raises(ValueError, match="FORMAL"):
        validate_bank({"kind": "smoke_only"})
    groups = []
    for task in range(20):
        for split in (("train", "val", "test") if task < 10 else ("test",)):
            groups.append({"task_id": task, "split": split,
                           "query_trajectory_ids": ["%s-query" % split] * 1000,
                           "context_trajectory_ids": ["%s-context%d" % (split, i) for i in range(20)]})
    manifest = {"kind": "formal_probe", "complete": True, "groups": groups}
    assert len(validate_probe_manifest(manifest)) == 40
    manifest["groups"][0]["context_trajectory_ids"][0] = "train-query"
    with pytest.raises(ValueError, match="overlap"):
        validate_probe_manifest(manifest)


def test_real_subprocess_queue_caps_gpus_and_continues_after_failure(project, monkeypatch):
    from component_ablations import scheduler, artifacts
    monkeypatch.setattr(scheduler, "check_gates", lambda root, phase: None)
    # This test verifies only scheduling. Scientific artifact checks have their
    # own tests; dummy process outputs stay under pytest's temporary directory.
    monkeypatch.setattr(artifacts, "verify_completion", lambda root, row, phase: read_json(scheduler.completion_path(root, row, phase)))
    events = project / "queue_events.jsonl"
    program = """
import json, os, pathlib, sys, time
marker, events, seed = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
def emit(kind):
    with events.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'kind':kind, 'time':time.time(), 'gpu':os.environ['CUDA_VISIBLE_DEVICES'], 'seed':seed}) + '\\n')
emit('start')
time.sleep(0.5)
emit('end')
if seed == 2:
    sys.exit(3)
marker.parent.mkdir(parents=True, exist_ok=True)
marker.write_text(json.dumps({'complete':True}), encoding='utf-8')
"""
    monkeypatch.setattr(scheduler, "make_command", lambda args, row, gpu: [sys.executable, "-c", program,
                        str(scheduler.completion_path(project, row, args.phase)), str(events), str(row["seed"])])
    args = SimpleNamespace(root=str(project), phase="smoke", envs=["cheetah-vel"], arms=["backbone"], seeds=[0, 1, 2, 3],
                           gpus=["2", "5"], jobs_per_gpu=1, max_parallel=2, dry_run=False, resume=False,
                           python=sys.executable, cpu=False, mujoco_root="/engines")
    assert scheduler.launch(args) == 1
    active, peak = {"2": 0, "5": 0}, 0
    for event in sorted([json.loads(line) for line in events.read_text().splitlines()], key=lambda item: item["time"]):
        active[event["gpu"]] += 1 if event["kind"] == "start" else -1
        assert all(0 <= value <= 1 for value in active.values())
        peak = max(peak, sum(active.values()))
        assert sum(active.values()) <= 2
    assert peak == 2
    jobs = read_json(project / "jobs_smoke.json")["jobs"]
    assert sum(job["status"] == "completed" for job in jobs) == 3
    assert sum(job["status"] == "failed" for job in jobs) == 1
    assert not (project / ".scheduler.lock").exists()


def test_frozen_plan_hashes_reject_silent_loss_or_config_edits(project):
    from component_ablations.artifacts import verify_plan
    verify_plan(project)
    path = project / "configs/cheetah-vel/no_real.json"
    config = read_json(path)
    config["algo_params"]["num_train_steps_per_itr"] = 401
    write_json(path, config)
    with pytest.raises(ValueError, match="Frozen arm"):
        verify_plan(project)


def test_resume_prunes_only_uncommitted_history_and_partial_tail(tmp_path):
    from component_ablations.worker import prune_history
    path = tmp_path / "history.jsonl"
    path.write_text('{"iteration":0,"value":4}\n{"iteration":1,"value":5}\n{"iteration":2,', encoding="utf-8")
    prune_history(path, 1)
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"iteration": 0, "value": 4}]
