"""Verify a returned audit and prepare a bounded source/config evidence request."""
import argparse
import collections
import json
from pathlib import Path, PurePosixPath

from audit import ENVIRONMENTS, PROTOCOL, PROTOCOL_ID, blank_models, sha256, write_json


SERVER_BASE = PurePosixPath('/root/tievnas2/YinCH')
SOURCE_ROOTS = ['GENTLE-consistant', 'FOCAL-v2', 'CORRO-native-v2',
                'backup/CSRO', 'UNICORN-ss-v2', 'ER-TRL-v2', 'backup/FOCAL']


def verify_audit(directory):
    completion = json.loads((directory / 'audit_completion.json').read_text(encoding='utf-8'))
    checks = {}
    for name, expected in completion['output_sha256'].items():
        if Path(name).name != name or name in ('.', '..') or '\\' in name:
            raise ValueError('Invalid audit member: ' + name)
        checks[name] = sha256(directory / name) == expected
    required = {'resource_audit.json', 'candidates.json', 'source_snapshots.json', 'models.json', 'protocol.json'}
    if not required.issubset(checks) or not all(checks.values()):
        raise ValueError('Incomplete or changed audit: ' + repr(checks))
    return checks


def is_diagnostic_run(path):
    return any(p.lower().startswith(('smoke', 'test_only', 'test-only', 'validation', 'acceptance'))
               for p in PurePosixPath(path).parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    checks = verify_audit(args.audit_dir)
    candidates = json.loads((args.audit_dir / 'candidates.json').read_text(encoding='utf-8'))
    report = json.loads((args.audit_dir / 'resource_audit.json').read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=False)
    roots = [str(SERVER_BASE / r) for r in SOURCE_ROOTS
             if any(c['path'].startswith(str(SERVER_BASE / r) + '/') for c in candidates)]
    targets = [c for c in candidates if any(e in PurePosixPath(c['path']).parts for e in ENVIRONMENTS)]
    # User-selected complete runs only; other GENTLE variants are not compared.
    targets = [c for c in targets if '/GENTLE-consistant/logs/' not in c['path']
               or any(label in PurePosixPath(c['path']).parts for label in PROTOCOL['run_selection'].values())]
    metadata = [c for c in targets if c['kind'] == 'shared_or_run_metadata'
                and (any(c['path'].startswith(r + '/') for r in roots)
                     or c['path'].startswith(str(SERVER_BASE / 'shared_protocol/shared-main-v1') + '/'))]
    shared = [c for c in metadata if '/shared_protocol/shared-main-v1/' in c['path']]
    checkpoints = [c for c in targets if c['kind'] == 'checkpoint_candidate'
                   and any(c['path'].startswith(r + '/') for r in roots)
                   and not is_diagnostic_run(c['path'])]
    groups = collections.defaultdict(list)
    for c in checkpoints:
        groups[str(PurePosixPath(c['path']).parent)].append(c)
    gentle_groups = {}
    for env in ENVIRONMENTS:
        variants = collections.defaultdict(set)
        for c in checkpoints:
            p = PurePosixPath(c['path'])
            if '/GENTLE-consistant/logs/' + env + '/' in c['path'] and p.name == 'policy_itr_499.pth':
                variants[p.parent.name].add(p.parent.parent.name)
        gentle_groups[env] = {k: sorted(v) for k, v in sorted(variants.items())}
    er = {env: [c for c in checkpoints if '/ER-TRL-v2/' in c['path'] and env in PurePosixPath(c['path']).parts]
          for env in ENVIRONMENTS}
    request = {
        'protocol': PROTOCOL_ID, 'protocol_configuration': PROTOCOL,
        'audit_directory_name': args.audit_dir.name,
        'audit_candidates_sha256': sha256(args.audit_dir / 'candidates.json'),
        'server_base': str(SERVER_BASE), 'source_roots': roots,
        'metadata_files': metadata,
        'checkpoint_inventory': checkpoints,
        'note': 'Discovery evidence only. No method assignment or stage completion. Do not copy weights.',
    }
    write_json(args.out / 'evidence_request.json', request)
    models = blank_models()
    by_path = {c['path']: c for c in candidates}
    for model in models['models']:
        label = model.get('user_selected_run_directory')
        if label is None:
            continue
        run = SERVER_BASE / 'GENTLE-consistant/logs' / model['environment'] / 'gentle' / ('seed' + str(model['training_seed'])) / label
        files = [by_path.get(str(run / (kind + '_itr_' + str(model['iteration']) + '.pth')))
                 for kind in ('context_encoder', 'policy')]
        model['candidate_run_directory'] = str(run)
        model['checkpoint_files'] = [f for f in files if f]
        model['training_config'] = by_path.get(str(run / 'variant.json'))
        model['source_root'] = str(SERVER_BASE / 'GENTLE-consistant')
        model['full_method_evidence'] = {'type': 'explicit user selection', 'configuration_verified': False}
        model['status'] = 'candidate_files_found_requires_validation' if all(files) else 'missing'
    write_json(args.out / 'models_unverified.json', models)
    write_json(args.out / 'protocol.json', PROTOCOL)
    review = {
        'protocol': PROTOCOL_ID, 'user_selected_runs': PROTOCOL['run_selection'],
        'audit_integrity': checks, 'stage_A_complete': False,
        'server_python': report['python_executable'],
        'server_packages': report['installed_package_metadata'],
        'shared_files': shared, 'gentle_variant_seed_coverage': gentle_groups,
        'ER_TRL_non_smoke_checkpoint_candidates': er,
        'candidate_run_directories': {k: v for k, v in sorted(groups.items())},
        'handoff_document': report.get('handoff_document'),
        'unresolved': ['User selected Ours-local=23_local_decay, Ours-global=28_global_decay and GENTLE=0_gentle_base; configuration/preprocessing verification remains pending',
                       'last.pt/final.pt iteration and full-method provenance require inspection',
                       'Source/config bodies are missing from the audit; request an evidence bundle',
                       'User confirmed ER-TRL cheetah-vel is still training; do not substitute smoke'],
    }
    write_json(args.out / 'review.json', review)
    lines = ['# Server audit review', '', 'Stage A: IN_PROGRESS. Real smoke: NOT_RUN.', '',
             'Integrity: all {} audited files match SHA256.'.format(len(checks)),
             'Relevant metadata files requested: {}.'.format(len(metadata)),
             'Source roots requested: {}.'.format(len(roots)), '',
             'Shared task manifests and normalizers exist for both target environments.',
             'GENTLE seed0/0_gentle_base has candidate iteration 200/499 files for both environments;',
             'its identity and preprocessing are not yet certified.', '',
             'ER-TRL cheetah-vel: only smoke candidates in this audit; user confirmed training in progress.',
             'Ours-local=23_local_decay; Ours-global=28_global_decay; GENTLE=0_gentle_base.',
             'Protocol v2 has 8 complete methods, 64 evaluation models, 66 total slots.', '',
             'Next: export source/config bodies with export_evidence.py. No full audit rerun needed.']
    (args.out / 'REVIEW.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Verified {} audit files; requested {} metadata files and {} source roots.'.format(
        len(checks), len(metadata), len(roots)))


if __name__ == '__main__':
    main()
