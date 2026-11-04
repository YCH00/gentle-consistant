# component-ablations-v1 — STATUS

Updated (UTC): 2026-11-04T06:37:21.967827+00:00

## A

RESOURCE_BLOCKED; independent implementation/testing continues

## B

{
  "status": "28_UNIT_TESTS_PASSED; real MuJoCo smoke pending",
  "checks": "smoke/unit_checks.json",
  "source_digest": "4adaa737be265c4260e1672664d195f311616f754c0e9a3f894e54e3b9f0ca55"
}

## C

{
  "status": "NOT_STARTED_RESOURCE_BLOCKED",
  "core_completed": 0,
  "core_total": 48,
  "supplementary_completed": 0,
  "supplementary_total": 16,
  "experiment_ids": "40-47",
  "name": "exp44"
}

## D

{
  "status": "IMPLEMENTED_INTERFACE; REAL_DATA_END_TO_END_VALIDATION_PENDING",
  "formal_fixed_context_bank_available": false,
  "formal_probe_queries_available": false,
  "main_completed": 0,
  "fixed_completed": 0,
  "probe_completed": 0,
  "resource_contract": "../component_ablations/RESOURCE_SCHEMAS.md"
}

## E

{
  "status": "REPORTING_IMPLEMENTED; AWAITING_REAL_FOUR_SEED_RESULTS",
  "summary_directory": "summaries/",
  "missing_cells": "incomplete; mean/std null",
  "performance_figures_generated": 0
}

## decisions

{
  "virtual_bc": true,
  "retrain_matched_full": true,
  "numbering": [
    {
      "arm": "backbone",
      "name": "40_exp44_backbone",
      "id": 40,
      "masks": {
        "R": 0,
        "V": 0,
        "P": 0
      }
    },
    {
      "arm": "real_only",
      "name": "41_exp44_real_only",
      "id": 41,
      "masks": {
        "R": 1,
        "V": 0,
        "P": 0
      }
    },
    {
      "arm": "no_real",
      "name": "42_exp44_no_real",
      "id": 42,
      "masks": {
        "R": 0,
        "V": 1,
        "P": 1
      }
    },
    {
      "arm": "no_virtual_recovery",
      "name": "43_exp44_no_virtual_recovery",
      "id": 43,
      "masks": {
        "R": 1,
        "V": 0,
        "P": 1
      }
    },
    {
      "arm": "no_virtual_rl",
      "name": "44_exp44_no_virtual_rl",
      "id": 44,
      "masks": {
        "R": 1,
        "V": 1,
        "P": 0
      }
    },
    {
      "arm": "full",
      "name": "45_exp44_full",
      "id": 45,
      "masks": {
        "R": 1,
        "V": 1,
        "P": 1
      }
    },
    {
      "arm": "full_global",
      "name": "46_exp44_full_global",
      "id": 46,
      "masks": {
        "R": 1,
        "V": 1,
        "P": 1
      }
    },
    {
      "arm": "full_constant",
      "name": "47_exp44_full_constant",
      "id": 47,
      "masks": {
        "R": 1,
        "V": 1,
        "P": 1
      }
    }
  ]
}

## validation

{
  "passed": true,
  "unit_tests": 28,
  "source_digest": "4adaa737be265c4260e1672664d195f311616f754c0e9a3f894e54e3b9f0ca55",
  "test_log": "smoke/unit-checks.log",
  "real_training": "Not run locally; server must rerun"
}

## blockers

[
  "2000 offline data files, 16 encoder/decoder weights and 80 task dynamics files are missing locally",
  "gym and gtimer missing in the local test interpreter; MuJoCo backend and torch/NumPy compatibility require validation on training host",
  "No complete formal v3 fixed context bank or frozen probe query manifest; old smoke contexts are not accepted"
]

## commands

[
  "python -m component_ablations plan --root component-ablations-v1 --start-id 40 --name exp44 --suite all --gpus 0,1,2,3 --jobs-per-gpu 1",
  "Anaconda3/python.exe -m component_ablations audit --root component-ablations-v1 [expected resource failure]",
  "Anaconda3/python.exe -m component_ablations check --root component-ablations-v1 [21 passed]",
  "python -m component_ablations launch --root component-ablations-v1 --phase train --gpus 2,3 --jobs-per-gpu 1 --max-parallel 2 --dry-run [64 slots]",
  "python -m component_ablations launch --root component-ablations-v1 --phase train --gpus 2,3 [expected resource gate failure]",
  "python -m component_ablations plot --root component-ablations-v1 [incomplete summaries; no plots]"
]

## resource_audit

{
  "resource_count": 2100,
  "verified_count": 4,
  "missing_or_mismatch_count": 2096,
  "dependencies": {
    "torch": true,
    "numpy": true,
    "gym": false,
    "gtimer": false,
    "click": true,
    "matplotlib": true,
    "sklearn": true,
    "tensorboard": true
  },
  "resources_ready": false
}

## next_steps

On restored server in gentle environment, upload/unzip component-io-recovery.zip. Preview with python recover_component_runs.py --root component-ablations-v1 --gpus 0,1,2 --jobs-per-gpu 4 --max-parallel 12 --group-seeds --dry-run. Remove --dry-run to resume eligible tasks. Read BLOCKED reasons and recovery plan for any unrecoverable attempts.

## summaries

Generated from actual records only; missing cells explicitly incomplete

## scope

Three-component R/V/P ablations. User-approved virtual critic + actor-Q + actor-BC. Core decay-only, no reconstruction/cycle weights or nearest-distance logging; constant schedule only in separately identified supplementary control.

## deployment_issue

{
  "host": "autodl-container-0d754ca1ec-ce8cd875",
  "reported_audit_verified": 2098,
  "expected_resources": 2100,
  "root_cause": "Windows CRLF task JSON files converted to LF during transfer. Both reported actual hashes exactly match historical LF task-manifest hashes; JSON payloads match frozen evidence.",
  "permanent_fix": "write_json uses explicit LF for create and replace; .gitattributes preserves frozen experiment bytes",
  "recovery_verified": "Simulated Linux checkout for 20 pinned task/reference/arm JSON files; restored original bytes using expected SHA256, with no hash-record edits",
  "server_recovery": "Command supplied to user; actual rerun pending",
  "local_training_resources": "Still unavailable; server audit observations are not local resource verification"
}

## deployment_evidence_fix

{
  "reported_server_result": "2100/2100 resources and all dependencies verified; 16 tests passed, 6 setup errors",
  "root_cause": "Portable planning_evidence omitted banks/<environment>/manifest.json for both environments",
  "backend_provenance_verified": "Existing resources.json backend_reference equals the original audited backend for both environments",
  "fix": "Added backend_reference_only manifests; planner bundles all 10 required evidence files and checks missing files before copying",
  "test_result": "23 passed with bundled evidence, including nested plan recreation without a historical directory",
  "server_action": "Run supplied metadata reconstruction command, then rerun commands.sh with GPUs 0,1,2; server rerun pending"
}

## scheduling

{
  "mode": "group-seeds",
  "group_key": [
    "environment",
    "arm"
  ],
  "seeds_per_group": [
    0,
    1,
    2,
    3
  ],
  "gpus": [
    "0",
    "1",
    "2"
  ],
  "jobs_per_gpu": 4,
  "max_parallel": 12,
  "rule": "One group per GPU. Wait for every seed to finish before taking another group; do not split a group across GPUs.",
  "script": "bash component-ablations-v1/commands.sh component-ablations-v1 0,1,2 4 group-seeds",
  "preview": "smoke/grouped_launch_preview.json",
  "verified": "Real subprocess queue tests for group affinity, per-GPU/global caps, and uneven seed completion",
  "deployment_archive": "../.temp/grouped-seeds-update.zip",
  "compatibility": "Source fingerprints include scheduler/planner/tests. Existing initialized or trained runs require compatibility handling before code replacement; active processes do not adopt new scheduling."
}

## restart_decision

{
  "authority": "User explicitly requested deleting existing server run records and restarting; supersedes checkpoint-resume request",
  "operation": "reset-runs",
  "server_root": "/root/tievnas2/YinCH/GENTLE-consistant/component-ablations-v1",
  "protected": "Frozen configs, experiment names/IDs, resource references and pretrained hashes, task/normalizer resources, planning evidence",
  "deleted_on_server_when_invoked": "Runs, initialization, smoke, logs, failures, summaries, figures and run/audit state",
  "tests": "Fresh-state cleanup preserves all frozen bytes, resets all 64 slots, rejects active workers and paths outside the experiment",
  "local_action": "Dry-run only on real workspace; destructive tests used pytest temporary directories",
  "server_action": "Pending user invocation; no remote deletion performed",
  "package": "../.temp/grouped-seeds-fresh-start.zip",
  "new_schedule": "GPUs 0,1,2; four seeds per environment/arm grouped on one GPU; max 12 processes"
}

## io_recovery

{
  "user_report": "2026-11-04: training scheduler and several workers failed with OSError Errno 5; exact storage path not present in supplied scheduler output",
  "decision": "Resume the existing experiment after storage recovery; supersedes earlier fresh-start reset instruction for this incident",
  "entry": "recover_component_runs.py",
  "archive": "../.temp/component-io-recovery.zip",
  "training_source_fingerprint_preserved": true,
  "source_digest": "4adaa737be265c4260e1672664d195f311616f754c0e9a3f894e54e3b9f0ca55",
  "behavior": "Verify completed results, select committed complete state or validated older slot, resume only tasks with checkpoints, start untouched tasks, block damaged/uncheckpointed attempts while preserving their files",
  "fallback_checks": "Pinned SHA256 where available, ZIP CRC integrity, complete-state format, source/config/initial identity, replay buffers, optimizer/network states, finite parameters, iteration counters, schedule and RNG",
  "records": "recovery/<timestamp>/plan.json and backed-up replaced pointers/obsolete lock",
  "tests": "10 recovery regression tests passed; includes bitwise matching next update after real tiny-model state restoration",
  "validation_limit": "Actual server checkpoint files and repaired storage have not been inspected locally"
}
