# STATUS — shift-diagnostics-v1

- A: IN_PROGRESS — resource discovery finished; assignments/adapters/smoke pending.
- B / C / D: NOT_STARTED.
- Only seven complete methods; no component ablations; one shared raw context bank.

## Executed command
```sh
/root/miniconda3/envs/gentle/bin/python /root/tievnas2/YinCH/GENTLE-consistant/shift_diagnostics/audit.py --repo-root /root/tievnas2/YinCH/GENTLE-consistant --search-root /root/tievnas2/YinCH --search-root /root/tievnas2/YinCH/shared_protocol/shared-main-v1 --hash-checkpoints --out /root/tievnas2/YinCH/shift-diagnostics-v1/audits/A-20261102T110236Z-3639
```

## Outputs and validation
- Output directory: `/root/tievnas2/YinCH/shift-diagnostics-v1/audits/A-20261102T110236Z-3639`
- `resource_audit.json`: search coverage, missing roots, errors, versions and Git state.
- `candidates.json`: 12032 candidates; names do not establish method/seed/iteration identity.
- `source_snapshots.json`: source/config/XML hashes; no checkpoint deserialization.
- `models.json`: 58 unresolved slots (56 evaluation models + 2 collector-200 models).
- `protocol.json`: proposed fixed protocol; resource hashes remain unresolved.
- True-environment smoke: NOT_RUN; stage A is NOT complete.

## Next step
Review candidates and actual run configurations; assign all model slots and shared resources,
verify matching encoder/policy provenance and normalizer/task hashes, then implement native
adapters and run each method/environment on the SAME raw smoke contexts. Do not start B yet.
Append every later command, output path, validation evidence and next step here.
