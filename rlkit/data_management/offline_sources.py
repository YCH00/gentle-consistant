"""Manifest-backed mixed training inputs; legacy inputs remain supported."""
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(root in path.parents, 'Manifest path escapes data root: ' + relative)
    return path


def validate_bundle(root, artifacts, tasks, split, mixed=False):
    root, artifacts = Path(root), Path(artifacts)
    manifest = read_json(artifacts / 'dataset_manifest.json')
    task_manifest = read_json(artifacts / 'task_manifest.json')
    provenance = read_json(artifacts / 'preparation_provenance.json')
    validation = read_json(artifacts / 'validation.json')
    hashes = {name: sha256(artifacts / name) for name in (
        'dataset_manifest.json', 'task_manifest.json', 'observation_normalizer.npz')}
    require(validation['artifact_status'] == 'passed', 'Artifact validation did not pass')
    for name, digest in hashes.items():
        require(provenance['core_sha256'][name] == digest, 'Artifact hash mismatch: ' + name)
    require(manifest['task_manifest_sha256'] == hashes['task_manifest.json'], 'Task hash mismatch')
    require(manifest['normalizer_sha256'] == hashes['observation_normalizer.npz'], 'Normalizer hash mismatch')
    require(list(tasks) == task_manifest[split + '_tasks'], 'Task split differs from manifest')
    if mixed:
        require(manifest['protocol_id'] == 'shared-native-v3', 'Expected mixed v3 manifest')
        require(manifest['data_action_coordinates'] == 'normalized', 'Mixed actions must be normalized')
        require(manifest['terminal_mode'] == 'legacy_episode_end', 'Unexpected terminal mode')
    selected, seen, rows = [], set(), {str(t): 0 for t in tasks}
    for entry in manifest['files']:
        if mixed:
            require(entry['splits'] == ['train'] and entry['task_id'] in tasks, 'Non-training mixed entry')
            require(entry['behavior'] in ('random', 'medium', 'expert'), 'Unknown behavior')
        manifest_split = 'test' if split == 'eval' else split
        if manifest_split not in entry['splits'] or entry['task_id'] not in tasks:
            continue
        path = checked_path(root, entry['path'])
        require(path not in seen, 'Duplicate trajectory: ' + str(path))
        seen.add(path)
        require(path.parent.name == 'goal_idx%d' % entry['task_id'], 'Trajectory task mismatch')
        require(sha256(path) == entry['sha256'], 'Trajectory hash mismatch: ' + str(path))
        array = np.load(path, allow_pickle=True)
        require(array.shape == (entry['rows'], 4) and len(array) > 0, 'Invalid trajectory shape')
        require(len(array) <= task_manifest['horizon'], 'Trajectory exceeds horizon')
        for col, key in enumerate(('observation', 'action', 'reward', 'next_observation')):
            values = np.asarray([np.asarray(v).reshape(-1) for v in array[:, col]], dtype=float)
            require(values.shape == (len(array), entry['dimensions'][key]) and np.isfinite(values).all(),
                    'Invalid trajectory values: ' + str(path))
            if mixed and col == 1:
                require(np.max(np.abs(values)) <= 1.000001, 'Mixed action outside [-1, 1]')
        rows[str(entry['task_id'])] += len(array)
        selected.append((str(path), entry['task_id']))
    require(all(rows.values()), 'Empty task data')
    for task, count in rows.items():
        require(count == manifest['task_rows'][task], 'Task row count mismatch: ' + task)
    if mixed:
        require(validation['files'] == len(selected), 'Validation file count mismatch')
        require(validation['task_rows'] == rows, 'Validation row count mismatch')
        require({p.resolve() for p in root.glob('goal_idx*/trj_evalsample*.npy')} == seen,
                'Mixed trajectory file set differs from manifest')
        collections = provenance.get('collection_sha256', {})
        require(set(collections) == {str(t) for t in tasks}, 'Missing collection hashes')
        for task, digest in collections.items():
            path = root / ('goal_idx%s' % task) / 'collection.json'
            require(sha256(path) == digest, 'Collection hash mismatch')
            collection = read_json(path)
            records = {e['path']: e for e in collection['files']}
            entries = [e for e in manifest['files'] if str(e['task_id']) == task]
            require(len(records) == len(entries), 'Collection file count mismatch')
            for entry in entries:
                record = records.get(manifest['environment'] + '/' + entry['path'], {})
                for key in ('task_id', 'trajectory_index', 'behavior', 'checkpoint', 'checkpoint_sha256', 'seed', 'rows', 'sha256'):
                    require(key in entry and record.get(key) == entry[key], 'Collection source mismatch: ' + key)
    selected.sort(key=lambda item: (item[1], int(Path(item[0]).name.split('evalsample')[1].split('_')[0].split('.')[0])))
    return selected, hashes


def trajectory_paths(params, tasks, split='train'):
    if params.get('mixed_artifacts_dir'):
        mixed = split == 'train'
        root = params['data_dir'] if mixed else params['eval_data_dir']
        artifacts = params['mixed_artifacts_dir'] if mixed else params['eval_artifacts_dir']
        entries, _ = validate_bundle(root, artifacts, tasks, split, mixed)
        return [p for p, _ in entries], [t for _, t in entries]
    root = Path(params['data_dir'])
    epoch = params['train_epoch' if split == 'train' else 'eval_epoch']
    suffix = '*' if epoch is None else str(int(epoch))
    entries = [(str(p), task) for n in range(params['n_trj']) for task in tasks
               for p in sorted((root / ('goal_idx%d' % task)).glob('trj_evalsample%d_step%s.npy' % (n, suffix)))]
    require(all(any(t == task for _, t in entries) for task in tasks), 'Missing legacy trajectories')
    return [p for p, _ in entries], [t for _, t in entries]


def load_trajectory(path, params, split='train'):
    array = np.load(path, allow_pickle=True)
    if params.get('mixed_artifacts_dir') and split == 'eval':
        manifest = read_json(Path(params['eval_artifacts_dir']) / 'dataset_manifest.json')
        coordinates = manifest['data_action_coordinates']
        require(coordinates in ('normalized', 'physical'), 'Unknown historical action coordinates')
        if coordinates == 'physical':
            provenance = read_json(Path(params['eval_artifacts_dir']) / 'preparation_provenance.json')
            low, high = np.asarray(provenance['action_low']), np.asarray(provenance['action_high'])
            for row in array:
                row[1] = 2 * (np.asarray(row[1]) - low) / (high - low) - 1
    return array


def dynamics_path(variant, seed):
    root = Path(__file__).resolve().parents[2] / 'dynamics' / variant['env_name']
    if variant['algo_params'].get('mixed_artifacts_dir'):
        return root / 'mixed_rme_v3_gentle' / ('seed%d' % seed) / variant['experiment_name']
    return root / ('expert_seed%d' % seed)


def bind_tasks(env, variant):
    """Use the published task parameters instead of resampling task identities."""
    params = variant['algo_params']
    if not params.get('mixed_artifacts_dir'):
        return
    manifest = read_json(Path(params['mixed_artifacts_dir']) / 'task_manifest.json')
    require(manifest['environment'] == variant['env_name'], 'Environment mismatch')
    require(manifest['horizon'] == params['max_path_length'], 'Horizon mismatch')
    tasks = manifest['tasks']
    require(len(tasks) == variant['env_params']['n_tasks'], 'Environment task count mismatch')
    dataset = read_json(Path(params['mixed_artifacts_dir']) / 'dataset_manifest.json')
    for entry in dataset['files']:
        dims = entry['dimensions']
        require(dims == dict(observation=int(np.prod(env.observation_space.shape)),
                             action=int(np.prod(env.action_space.shape)), reward=1,
                             next_observation=int(np.prod(env.observation_space.shape))),
                'Dataset dimensions differ from environment')
    if variant['env_name'] == 'point-robot':
        env.goals = [np.asarray(t) for t in tasks]
    else:
        env.tasks = [{k: np.asarray(v) if isinstance(v, list) else v for k, v in t.items()} for t in tasks]
    env.reset_task(0)
