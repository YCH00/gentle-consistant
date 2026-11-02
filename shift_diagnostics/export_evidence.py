"""Export audited source/config evidence only. Never copy or load RL weights."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import zipfile

from audit import SKIP_DIRS, git_info, sha256, write_json
from review_audit import verify_audit


SOURCE_SKIP = SKIP_DIRS | {
    'logs', 'data', 'datasets', 'dynamics', 'encoder_decoder', 'runs', 'results',
    'result', 'output', 'artifacts', 'checkpoints', 'checks', 'figures', 'reproduction',
}
SOURCE_SUFFIXES = {'.py', '.json', '.yaml', '.yml', '.xml', '.sh', '.bash', '.md'}
NORMALIZERS = {'observation_normalizer.npz', 'obs_normalizer.npz'}


def inside(path, root):
    return path == root or root in path.parents


def approved_path(path, base):
    """Reject traversal/symlink escapes; never follow source-tree symlinks."""
    path = Path(path)
    resolved = path.resolve()
    if not inside(resolved, base):
        raise ValueError('Path outside server base: ' + str(path))
    relative = path.absolute().relative_to(base)
    cursor = base
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError('Symlink requires explicit review: ' + str(path))
    return resolved


def collect_sources(root, errors):
    for current, dirs, files in os.walk(str(root), followlinks=False,
                                        onerror=lambda exc: errors.append(str(exc))):
        current = Path(current)
        dirs[:] = sorted(d for d in dirs if d not in SOURCE_SKIP and not d.startswith('.')
                         and not (current / d).is_symlink())
        for name in sorted(files):
            path = current / name
            if path.is_symlink() or name.startswith('.'):
                continue
            if path.suffix.lower() in SOURCE_SUFFIXES or name.startswith('requirements') or name == 'requrements.txt':
                yield path


def export(request, audit_dir, out, max_file_bytes=8 * 1024 * 1024,
           max_total_bytes=128 * 1024 * 1024):
    verify_audit(audit_dir)
    if sha256(audit_dir / 'candidates.json') != request['audit_candidates_sha256']:
        raise ValueError('Request belongs to a different audit')
    base = Path(request['server_base']).resolve()
    if not base.is_dir():
        raise ValueError('Run on the server; base not found: ' + str(base))
    if out.exists():
        raise ValueError('Refusing to overwrite: ' + str(out))
    out.mkdir(parents=True, exist_ok=False)
    candidates = json.loads((audit_dir / 'candidates.json').read_text(encoding='utf-8'))
    audited = {c['path']: c for c in candidates}
    records, errors, selected, roots = [], [], {}, []
    for value in request['source_roots']:
        root = approved_path(value, base)
        if not root.is_dir():
            errors.append('Missing source root: ' + str(root))
            continue
        roots.append(root)
        for path in collect_sources(root, errors):
            selected[path] = ('source', None)
    for item in request['metadata_files']:
        path = approved_path(item['path'], base)
        # Only audited JSON and normalizer files can be exported as run metadata.
        if item['path'] not in audited or audited[item['path']]['kind'] != 'shared_or_run_metadata':
            raise ValueError('Metadata was not recorded in audit: ' + item['path'])
        if path.suffix != '.json' and path.name not in NORMALIZERS:
            raise ValueError('Disallowed metadata extension: ' + str(path))
        selected[path] = ('metadata', audited[item['path']].get('sha256'))
    total = 0
    archive_path = out / 'evidence.zip'
    with zipfile.ZipFile(str(archive_path.with_suffix('.zip.partial')), 'x', zipfile.ZIP_DEFLATED) as archive:
        for path, (kind, previous_hash) in sorted(selected.items(), key=lambda item: str(item[0])):
            path = approved_path(path, base)
            record = {'source_path': str(path), 'kind': kind, 'audit_sha256': previous_hash}
            try:
                size = path.stat().st_size
                if size > max_file_bytes or total + size > max_total_bytes:
                    raise ValueError('Export size budget exceeded')
                with path.open('rb') as stream:
                    payload = stream.read(max_file_bytes + 1)
                if len(payload) > max_file_bytes or total + len(payload) > max_total_bytes:
                    raise ValueError('Export size budget exceeded during read')
                member = 'files/' + path.relative_to(base).as_posix()
                digest = hashlib.sha256(payload).hexdigest()
                archive.writestr(member, payload)
                total += len(payload)
                record.update(member=member, sha256=digest, size_bytes=len(payload), status='exported',
                              matches_audit=(digest == previous_hash) if previous_hash else None)
            except (OSError, ValueError) as exc:
                record.update(status='not_exported', error=str(exc))
                errors.append(str(path) + ': ' + str(exc))
            records.append(record)
        manifest = {'protocol': request.get('protocol', 'unspecified'), 'stage_A_complete': False,
                    'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'request_audit_sha256': request['audit_candidates_sha256'],
                    'files': records, 'errors': errors, 'uncompressed_bytes': total,
                    'source_repositories': [git_info(root) for root in roots],
                    'checkpoint_action': 'No checkpoint copied, loaded, or rehashed',
                    'changed_metadata': [r['source_path'] for r in records if r.get('matches_audit') is False]}
        archive.writestr('bundle_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr('checkpoint_inventory.json', json.dumps(request['checkpoint_inventory'], indent=2))
        archive.writestr('protocol.json', json.dumps(request.get('protocol_configuration', {}), indent=2))
    archive_path.with_suffix('.zip.partial').replace(archive_path)
    write_json(out / 'bundle_manifest.json', manifest)
    write_json(out / 'completion.json', {'evidence_export_finished': True, 'stage_A_complete': False,
                                         'all_requested_files_exported': not errors,
                                         'archive_sha256': sha256(archive_path), 'archive': archive_path.name})
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, default=Path(__file__).with_name('evidence_request.json'))
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding='utf-8'))
    if request.get('protocol') not in ('shift-diagnostics-v1', 'shift-diagnostics-v2'):
        raise ValueError('Unrecognized protocol output directory')
    out = args.out or (Path(request['server_base']) / request['protocol'] / 'evidence' /
                       ('A-evidence-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(os.getpid())))
    manifest = export(request, args.audit_dir, out)
    command = ' '.join(shlex.quote(x) for x in [sys.executable, str(Path(__file__).resolve())] +
                       (list(argv) if argv is not None else sys.argv[1:]))
    (out / 'commands.sh').write_text('#!/usr/bin/env bash\n# Actual invocation; rerun with a NEW output directory.\n' + command + '\n', encoding='utf-8')
    status = ('# STATUS — stage A evidence export\n\nA: IN_PROGRESS; B/C/D: NOT_STARTED.\n\n'
              'Executed command:\n```bash\n' + command + '\n```\n\nOutput: `' + str(out) + '`\n\n'
              'Exported files: ' + str(sum(r['status'] == 'exported' for r in manifest['files'])) + '\n\n'
              'Errors: ' + str(len(manifest['errors'])) + '; changed metadata: ' + str(len(manifest['changed_metadata'])) + '\n\n'
              'No checkpoint was loaded; real smoke NOT_RUN.\n\n'
              'Next: download this directory, review source/config bodies, resolve model identity and iteration, then implement smoke.\n')
    (out / 'STATUS.md').write_text(status, encoding='utf-8')
    print('Evidence directory: ' + str(out))
    print('Download this directory. Files={}, errors={}, changed_metadata={}. Stage A is not complete.'.format(
        len(manifest['files']), len(manifest['errors']), len(manifest['changed_metadata'])))
    return 0 if not manifest['errors'] else 2


if __name__ == '__main__':
    sys.exit(main())
