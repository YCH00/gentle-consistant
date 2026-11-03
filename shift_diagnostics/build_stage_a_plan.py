"""Build a source-pinned stage-A plan from a verified downloaded evidence bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from audit import PROTOCOL, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Plan output already exists')
    completion = json.loads((args.evidence_dir / 'completion.json').read_text(encoding='utf-8'))
    if sha256(args.evidence_dir / 'evidence.zip') != completion['archive_sha256']:
        raise ValueError('Evidence ZIP checksum mismatch')
    base = '/root/tievnas2/YinCH'
    sources = {'GENTLE': base + '/GENTLE-consistant', 'Ours-local': base + '/GENTLE-consistant',
               'Ours-global': base + '/GENTLE-consistant', 'FOCAL': base + '/FOCAL-v2',
               'CORRO': base + '/CORRO-native-v2', 'CSRO': base + '/backup/CSRO',
               'UNICORN-SS': base + '/UNICORN-ss-v2', 'ER-TRL': base + '/ER-TRL-v2'}
    plan = {'protocol': PROTOCOL, 'server_base': base, 'reference_root': sources['GENTLE'],
            'sources': sources, 'shared': {}, 'source_files': [], 'evidence_sha256': completion['archive_sha256']}
    with zipfile.ZipFile(args.evidence_dir / 'evidence.zip') as archive:
        manifest = json.loads(archive.read('bundle_manifest.json'))
        if manifest['errors'] or manifest['changed_metadata']:
            raise ValueError('Evidence has unresolved errors/metadata changes')
        by_path = {}
        for row in manifest['files']:
            payload = archive.read(row['member'])
            if hashlib.sha256(payload).hexdigest() != row['sha256']:
                raise ValueError('Evidence member checksum mismatch')
            by_path[row['source_path']] = row
            # Pin the actual method/environment code, excluding diagnostic tools
            # and optional tests/docs which do not define model behavior.
            path = row['source_path']
            selected_source = any(path.startswith(root + '/') for root in set(sources.values()))
            selected_legacy_config = any(path == sources['GENTLE'] + '/logs/' + env + '/gentle/seed' + str(seed) + '/' + run + '/variant.json'
                for env in PROTOCOL['environments'] for seed in PROTOCOL['training_seeds'] for run in PROTOCOL['run_selection'].values())
            if (selected_source and path.endswith('.py') and '/shift_diagnostics/' not in path and '/tests/' not in path) or selected_legacy_config:
                plan['source_files'].append({'path': path, 'sha256': row['sha256']})
        for env in PROTOCOL['environments']:
            shared = {}
            for name in ('task_manifest.json', 'dataset_manifest.json', 'observation_normalizer.npz', 'preparation_provenance.json'):
                path = base + '/shared_protocol/shared-main-v1/' + env + '/' + name
                row = by_path[path]
                shared[name] = {'path': path, 'sha256': row['sha256']}
            shared['task_payload'] = json.loads(archive.read(by_path[shared['task_manifest.json']['path']]['member']))
            shared['preparation_payload'] = json.loads(archive.read(by_path[shared['preparation_provenance.json']['path']]['member']))
            plan['shared'][env] = shared
    write_json(args.out, plan)
    print('Stage-A source-pinned plan written:', args.out)


if __name__ == '__main__':
    main()
