"""Verify collected mixed trajectories and publish one environment's v3 bundle."""

import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys
import tempfile

from collect_mixed_rme import ENVIRONMENTS, PROTOCOL


def export(args):
    import numpy as np

    sys.path.insert(1, str(args.focal_root.resolve()))
    from focal_v2.artifacts import exported_statistics, read_trajectory
    from focal_v2.common import benchmark, file_hash, read_json, write_json

    cfg = benchmark(args.env)
    old = args.source_root / args.env
    tasks_file = old / 'task_manifest.json'
    tasks_hash = file_hash(tasks_file)
    tasks = read_json(tasks_file)
    original = read_json(old / 'dataset_manifest.json')
    original_provenance = read_json(old / 'preparation_provenance.json')
    if (tasks.get('environment') != args.env or tasks.get('train_tasks') != cfg['train_tasks']
            or tasks.get('eval_tasks') != cfg['eval_tasks'] or tasks.get('horizon') != cfg['horizon']
            or original.get('environment') != args.env
            or original.get('task_manifest_sha256') != tasks_hash):
        raise ValueError('Historical bundle does not match the environment or task manifest')
    dimensions = original['files'][0]['dimensions']
    behavior_names = ('random', 'medium', 'expert')
    quotas = dict(zip(behavior_names, (34, 33, 33) if cfg['n_trj'] == 100 else (17, 17, 16)))
    files, observations = [], []
    task_rows, behavior_rows, collection_sha256 = {}, {}, {}
    seeds = set()
    for task in cfg['train_tasks']:
        folder = Path(args.env) / f'goal_idx{task}'
        collection_path = args.raw_root / folder / 'collection.json'
        collection = read_json(collection_path)
        expected = dict(protocol_id=PROTOCOL, environment=args.env, task_id=task,
                        task_manifest_sha256=tasks_hash, data_action_coordinates='normalized',
                        action_low=original_provenance['action_low'],
                        action_high=original_provenance['action_high'])
        if any(collection.get(key) != value for key, value in expected.items()):
            raise ValueError(f'{folder}: collection identity mismatch')
        backend = collection['backend']
        for key in ('reference_commit', 'reference_sources_sha256', 'class_name', 'placeholder'):
            if backend[key] != original_provenance['backend'][key]:
                raise ValueError(f'{folder}: backend mismatch for {key}')
        if (backend.get('xml') or {}).get('sha256') != (
                original_provenance['backend'].get('xml') or {}).get('sha256'):
            raise ValueError(f'{folder}: XML mismatch')
        entries = collection['files']
        names = {f'trj_evalsample{index}.npy' for index in range(cfg['n_trj'])}
        if (len(entries) != len(names) or {Path(e['path']).name for e in entries} != names
                or {p.name for p in collection_path.parent.glob('*.npy')} != names
                or Counter(e['behavior'] for e in entries) != quotas):
            raise ValueError(f'{folder}: unexpected trajectory set or behavior quotas')
        if not 0.35 <= collection['medium_fraction'] <= 0.65:
            raise ValueError(f'{folder}: Medium calibration outside registered range')
        task_counts = Counter()
        calibration_seeds = set(collection['calibration_seeds'])
        for entry in entries:
            filename = f'trj_evalsample{entry["trajectory_index"]}.npy'
            path = collection_path.parent / filename
            if (entry['task_id'] != task or entry['path'] != (folder / filename).as_posix()
                    or path.is_symlink() or file_hash(path) != entry['sha256']):
                raise ValueError(f'{path}: trajectory identity mismatch')
            behavior, checkpoint = entry['behavior'], entry['checkpoint']
            if behavior == 'random':
                valid_source = checkpoint is None and entry['checkpoint_sha256'] is None
            else:
                checkpoint_hash = entry['checkpoint_sha256']
                valid_source = (isinstance(checkpoint_hash, str) and len(checkpoint_hash) == 64
                                and isinstance(checkpoint, int) and (
                                    0 < checkpoint < cfg['data_checkpoint'] if behavior == 'medium'
                                    else checkpoint == cfg['data_checkpoint']))
            if not valid_source or entry['seed'] in calibration_seeds or entry['seed'] in seeds:
                raise ValueError(f'{path}: invalid behavior source or repeated collection seed')
            seeds.add(entry['seed'])
            batch = read_trajectory(path, dimensions['observation'], dimensions['action'], cfg['horizon'])
            rows = len(batch['observations'])
            if (entry['rows'] != rows or np.max(np.abs(batch['actions'])) > 1 + 1e-6
                    or not np.isclose(batch['rewards'].sum(), entry['episode_return'])):
                raise ValueError(f'{path}: row count, normalized actions or episode return mismatch')
            files.append(dict(entry, path=(folder.relative_to(args.env) / filename).as_posix(),
                              splits=['train'], dimensions=dimensions))
            observations.append(batch['observations'])
            task_counts[behavior] += rows
        task_rows[str(task)] = sum(task_counts.values())
        behavior_rows[str(task)] = dict(task_counts)
        collection_sha256[str(task)] = file_hash(collection_path)

    destination = args.output_root / args.env
    if destination.exists():
        raise FileExistsError(f'Bundle already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.publish-', dir=destination.parent) as temporary:
        stage = Path(temporary) / 'bundle'
        stage.mkdir()
        shutil.copyfile(tasks_file, stage / 'task_manifest.json')
        np.savez(stage / 'observation_normalizer.npz',
                 **exported_statistics(np.concatenate(observations)),
                 format_version=1, train_tasks=cfg['train_tasks'], eval_tasks=cfg['eval_tasks'])
        write_json(stage / 'dataset_manifest.json', dict(
            protocol_id=PROTOCOL, environment=args.env, data_action_coordinates='normalized',
            observation_coordinates='raw', reward_coordinates='raw', terminal_mode='legacy_episode_end',
            task_manifest_sha256=tasks_hash, normalizer_sha256=file_hash(stage / 'observation_normalizer.npz'),
            task_rows=task_rows, behavior_rows=behavior_rows, behavior_trajectory_quotas=quotas,
            expert_only_task_rows={task: original['task_rows'][task] for task in task_rows},
            files=files,
        ))
        write_json(stage / 'preparation_provenance.json', dict(
            protocol_id=PROTOCOL, environment=args.env, collection_sha256=collection_sha256,
            old_dataset_manifest_sha256=file_hash(old / 'dataset_manifest.json'),
            action_coordinates='normalized', action_low=original_provenance['action_low'],
            action_high=original_provenance['action_high'], backend=original_provenance['backend'],
            core_sha256={name: file_hash(stage / name) for name in (
                'task_manifest.json', 'observation_normalizer.npz', 'dataset_manifest.json')},
        ))
        write_json(stage / 'validation.json', dict(
            artifact_status='passed', fingerprints='passed', train_only_statistics='passed',
            task_manifest='passed', files=len(files), task_rows=task_rows,
        ))
        stage.rename(destination)
    print(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', required=True, choices=ENVIRONMENTS)
    parser.add_argument('--focal-root', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True, help='Historical shared-main-v1 root')
    parser.add_argument('--output-root', type=Path, required=True, help='New shared-native-v3 root')
    export(parser.parse_args())
