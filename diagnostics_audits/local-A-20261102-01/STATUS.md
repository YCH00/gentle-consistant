# STATUS — shift-diagnostics-v1

- A: IN_PROGRESS — resource discovery finished; assignments/adapters/smoke pending.
- B / C / D: NOT_STARTED.
- Only seven complete methods; no component ablations; one shared raw context bank.

## Executed command
```sh
'C:\Python312\python.exe' 'D:\Desktop\offline meta RL\gentle-consistant\shift_diagnostics\audit.py' --repo-root . --hash-checkpoints --out diagnostics_audits/local-A-20261102-01
```

## Outputs and validation
- Output directory: `D:\Desktop\offline meta RL\gentle-consistant\diagnostics_audits\local-A-20261102-01`
- `resource_audit.json`: search coverage, missing roots, errors, versions and Git state.
- `candidates.json`: 2 candidates; names do not establish method/seed/iteration identity.
- `source_snapshots.json`: source/config/XML hashes; no checkpoint deserialization.
- `models.json`: 58 unresolved slots (56 evaluation models + 2 collector-200 models).
- `protocol.json`: proposed fixed protocol; resource hashes remain unresolved.
- True-environment smoke: NOT_RUN; stage A is NOT complete.

## Next step
Review candidates and actual run configurations; assign all model slots and shared resources,
verify matching encoder/policy provenance and normalizer/task hashes, then implement native
adapters and run each method/environment on the SAME raw smoke contexts. Do not start B yet.
Append every later command, output path, validation evidence and next step here.
