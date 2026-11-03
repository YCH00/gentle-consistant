"""Bind native final run groups before any smoke rewards are observed."""
import json
from pathlib import Path

from audit import PROTOCOL, sha256


def record(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha256(path), 'size_bytes': path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def final_record(run, method):
    path = run / ('final_checkpoint.json' if method == 'CORRO' else 'final.json')
    if not path.is_file():
        raise ValueError('Final completion marker missing: ' + str(path))
    final = read(path)
    if not final.get('completed_native_plan') or final.get('smoke'):
        raise ValueError('Not a completed native formal run')
    filename = final.get('checkpoint', final.get('checkpoint_file', final.get('file')))
    checkpoint = (run / filename).resolve()
    if run.resolve() not in checkpoint.parents or not checkpoint.is_file():
        raise ValueError('Invalid final checkpoint path')
    return path, final, checkpoint


def groups(plan, method, environment):
    root = Path(plan['sources'][method])
    if method == 'FOCAL':
        return [root / 'results/shared-native-v2/run01' / environment]
    if method == 'CORRO':
        return [root / 'runs/shared-native-v2-3windows' / environment]
    if method == 'CSRO':
        return [root / 'runs/CSRO' / environment]
    parent = root / 'runs/shared-native-v2' / environment
    return sorted(p for p in parent.glob('*') if p.is_dir() and not p.name.lower().startswith(('smoke', 'test')))


def select_models(plan):
    entries, collectors, unavailable, selection_log = [], [], [], []
    for environment in PROTOCOL['environments']:
        for method in PROTOCOL['methods']:
            legacy = method in PROTOCOL['run_selection']
            if legacy:
                group = Path(plan['reference_root']) / 'logs' / environment / 'gentle'
                run_paths = [group / ('seed' + str(seed)) / PROTOCOL['run_selection'][method] for seed in range(4)]
            else:
                complete_groups = []
                for group in groups(plan, method, environment):
                    try:
                        run_paths = [group / ('seed' + str(seed)) for seed in range(4)]
                        if method == 'CSRO':
                            if not all((run / 'checkpoints/itr_499.pth').is_file() for run in run_paths):
                                raise ValueError('CSRO four final checkpoints not present')
                            timestamp = max((run / 'checkpoints/itr_499.pth').stat().st_mtime_ns for run in run_paths)
                        else:
                            finals = [final_record(run, method) for run in run_paths]
                            timestamp = max(row[0].stat().st_mtime_ns for row in finals)
                        complete_groups.append((timestamp, str(group), run_paths))
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        selection_log.append({'method': method, 'environment': environment, 'group': str(group), 'eligible': False, 'reason': str(exc)})
                if not complete_groups:
                    unavailable.append({'method': method, 'environment': environment, 'status': 'pending_complete_four_seed_run'})
                    continue
                _, selected_group, run_paths = max(complete_groups, key=lambda row: (row[0], row[1]))
                selection_log.append({'method': method, 'environment': environment, 'group': selected_group,
                                      'eligible': True, 'rule': 'latest completed four-seed group, before diagnostic evaluation'})
            for seed, run in enumerate(run_paths):
                try:
                    config_path = run / ('variant.json' if legacy else 'resolved_config.json')
                    configuration = read(config_path)
                    configured_env = configuration.get('environment', configuration.get('env_name', configuration.get('native', {}).get('env_name')))
                    if configured_env != environment:
                        raise ValueError('Training configuration environment differs: ' + str(configured_env))
                    if legacy:
                        paths = [run / ('context_encoder_itr_499.pth'), run / 'policy_itr_499.pth']
                        final = None
                    elif method == 'CSRO':
                        paths, final = [run / 'checkpoints/itr_499.pth'], None
                    else:
                        _, final, path = final_record(run, method)
                        paths = [path]
                    print('Hashing {} {} seed{}'.format(environment, method, seed), flush=True)
                    records = [record(p) for p in paths]
                    if final is not None:
                        expected_hash = final.get('checkpoint_hash', final.get('sha256'))
                        if records[0]['sha256'] != expected_hash:
                            raise ValueError('Final file hash differs from completion marker')
                    entry = dict(method=method, environment=environment, training_seed=seed,
                                 final_iteration_zero_based=PROTOCOL['expected_final_iteration_zero_based'][method],
                                 source_root=plan['sources'][method], run_directory=str(run),
                                 configuration=configuration, config_file=record(config_path),
                                 checkpoint_files=records, native_final_marker=final,
                                 status='selected_pending_strict_load_and_smoke')
                    entries.append(entry)
                    if method == 'GENTLE' and seed == 0:
                        collector = dict(entry, final_iteration_zero_based=200, role='collector200')
                        collector['checkpoint_files'] = [record(run / name) for name in
                            ('context_encoder_itr_200.pth', 'policy_itr_200.pth')]
                        collectors.append(collector)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    unavailable.append({'method': method, 'environment': environment, 'training_seed': seed,
                                        'status': 'resource_error', 'reason': str(exc)})
    return {'protocol': PROTOCOL['protocol_id'], 'models': entries, 'collectors': collectors, 'unavailable': unavailable,
            'selection_log': selection_log, 'complete': False}
