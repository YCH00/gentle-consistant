"""Shared legacy/v3 trajectory selection and mixed-experiment provenance.

No torch or environment imports: validation can run without MuJoCo.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def mixed(params):
    mode = params.get('train_data_format', 'legacy')
    if mode not in ('legacy', 'mixed_rme_v3'):
        raise ValueError('Unknown train_data_format: ' + mode)
    return mode == 'mixed_rme_v3'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(root in path.parents, 'Manifest path escapes data root: ' + str(relative))
    return path


def task_id(path):
    return int(Path(path).parent.name[len('goal_idx'):])


def legacy_paths(root, epoch, n_trj, tasks):
    paths = []
    for task in tasks:
        selected = []
        for index in range(n_trj):
            suffix = '*' if epoch is None else str(int(epoch))
            selected.extend(sorted((Path(root) / ('goal_idx%d' % task)).glob(
                'trj_evalsample%d_step%s.npy' % (index, suffix))))
        require(bool(selected), 'No legacy trajectories for task %s under %s' % (task, root))
        paths.extend(str(path) for path in selected)
    return paths


def validate_trajectory(path, dimensions=None, rows=None, horizon=None, normalized=False):
    data = np.load(path, allow_pickle=True)
    require(data.ndim == 2 and data.shape[1] == 4 and len(data) > 0,
            'Expected nonempty (T, 4) trajectory: ' + str(path))
    require(rows is None or len(data) == rows, 'Row count mismatch: ' + str(path))
    require(horizon is None or len(data) <= horizon, 'Trajectory exceeds horizon: ' + str(path))
    columns = [np.asarray(list(data[:, i]), dtype=np.float64) for i in range(4)]
    require(all(np.isfinite(x).all() for x in columns), 'Non-finite trajectory: ' + str(path))
    if dimensions:
        obs, act = dimensions['observation'], dimensions['action']
        require(columns[0].shape == columns[3].shape == (len(data), obs), 'Observation shape mismatch')
        require(columns[1].shape == (len(data), act), 'Action shape mismatch')
    require(columns[2].shape in ((len(data),), (len(data), 1)), 'Reward shape mismatch')
    if normalized:
        require(np.max(np.abs(columns[1])) <= 1.000001, 'Action outside [-1, 1]: ' + str(path))
    return data


def bundle(params):
    directory = Path(params['train_artifacts_dir'])
    manifest = read_json(directory / 'dataset_manifest.json')
    tasks = read_json(directory / 'task_manifest.json')
    provenance = read_json(directory / 'preparation_provenance.json')
    validation = read_json(directory / 'validation.json')
    require(manifest['protocol_id'] == provenance['protocol_id'] == 'shared-native-v3', 'Wrong v3 protocol')
    require(tasks['protocol_id'] == 'shared-main-v1', 'Wrong task protocol')
    require(validation['artifact_status'] == 'passed', 'Dataset validation has not passed')
    for name in ('task_manifest.json', 'observation_normalizer.npz', 'dataset_manifest.json'):
        require(file_hash(directory / name) == provenance['core_sha256'][name], 'Artifact hash mismatch: ' + name)
    require(manifest['task_manifest_sha256'] == file_hash(directory / 'task_manifest.json'), 'Task hash mismatch')
    require(manifest['normalizer_sha256'] == file_hash(directory / 'observation_normalizer.npz'), 'Normalizer hash mismatch')
    require(manifest['environment'] == tasks['environment'] == provenance['environment'], 'Environment mismatch')
    for key, expected in [('data_action_coordinates', 'normalized'), ('observation_coordinates', 'raw'),
                          ('reward_coordinates', 'raw'), ('terminal_mode', 'legacy_episode_end')]:
        require(manifest[key] == expected, 'Unsupported ' + key)
    return manifest, tasks, provenance, validation


def training_paths(params, tasks):
    if not mixed(params):
        return legacy_paths(params['data_dir'], params['train_epoch'], params['n_trj'], tasks)
    manifest, task_manifest, provenance, validation = bundle(params)
    require(list(tasks) == task_manifest['train_tasks'], 'Training task split mismatch')
    root = Path(params['data_dir'])
    paths, seen, rows, counts = [], set(), Counter(), Counter()
    for entry in sorted(manifest['files'], key=lambda x: (x['task_id'], x['trajectory_index'])):
        task = entry['task_id']
        require(entry['splits'] == ['train'] and task in tasks, 'Unexpected split/task in v3 manifest')
        path = checked_path(root, entry['path'])
        require(entry['dimensions'] == manifest['files'][0]['dimensions'], 'Inconsistent trajectory dimensions')
        require(path not in seen and task_id(path) == task, 'Duplicate path or incorrect task directory')
        seen.add(path)
        require(file_hash(path) == entry['sha256'], 'Trajectory hash mismatch: ' + str(path))
        validate_trajectory(path, entry['dimensions'], entry['rows'], task_manifest['horizon'], True)
        rows[str(task)] += entry['rows']
        counts[(task, entry['behavior'])] += 1
        paths.append(str(path))
    require({path.resolve() for path in root.glob('goal_idx*/trj_evalsample*.npy')} == seen,
            'Mixed trajectory file set differs from manifest')
    capacity = params.get('replay_buffer_size')
    require(capacity is None or all(value <= capacity for value in rows.values()), 'Replay buffer would truncate dataset')
    require(dict(rows) == manifest['task_rows'] == validation['task_rows'], 'Task row totals mismatch')
    require(len(paths) == validation['files'], 'File count mismatch')
    require(set(rows) == {str(task) for task in tasks}, 'Missing training tasks')
    quotas = manifest['behavior_trajectory_quotas']
    for task in tasks:
        actual = {behavior: count for (idx, behavior), count in counts.items() if idx == task}
        require(actual == quotas, 'Behavior quota mismatch for task %s: %s' % (task, actual))
        require(sum(actual.values()) == params['n_trj'], 'n_trj disagrees with manifest')
        collection = root / ('goal_idx%d' % task) / 'collection.json'
        require(file_hash(collection) == provenance['collection_sha256'][str(task)], 'Collection hash mismatch')
    print('Validated mixed training data: %d tasks, %d trajectories, %d transitions' %
          (len(tasks), len(paths), sum(rows.values())))
    return paths


def evaluation_paths(params, tasks):
    root = params.get('eval_data_dir') or params['data_dir']
    paths = legacy_paths(root, params['eval_epoch'], params.get('eval_n_trj', params['n_trj']), tasks)
    if mixed(params):
        _, task_manifest, provenance, _ = bundle(params)
        require(list(tasks) == task_manifest['eval_tasks'], 'Evaluation task split mismatch')
        directory = Path(params['eval_artifacts_dir'])
        manifest_path = directory / 'dataset_manifest.json'
        require(file_hash(manifest_path) == provenance['old_dataset_manifest_sha256'], 'Historical dataset hash mismatch')
        require(file_hash(directory / 'task_manifest.json') == file_hash(Path(params['train_artifacts_dir']) / 'task_manifest.json'),
                'Training/evaluation task manifests differ')
        expected_count = params.get('eval_n_trj', params['n_trj'])
        require(all(sum(task_id(path) == task for path in paths) == expected_count for task in tasks),
                'Missing or extra historical evaluation trajectories')
        old = read_json(manifest_path)
        entries = {item['path']: item for item in old['files']}
        for path in paths:
            relative = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
            entry = entries[relative]
            require(entry['task_id'] == task_id(path), 'Historical task mismatch')
            require(file_hash(path) == entry['sha256'], 'Historical trajectory hash mismatch: ' + path)
            validate_trajectory(path, entry['dimensions'], entry['rows'], task_manifest['horizon'])
    return paths


def experiment_dir(variant, kind, seed):
    tag = variant['algo_params'].get('dataset_tag', 'mixed_rme_v3' if mixed(variant['algo_params']) else 'expert')
    require(tag and Path(tag).name == tag and tag not in ('.', '..'), 'dataset_tag must be a directory name')
    return ROOT / kind / variant['env_name'] / ('%s_seed%s' % (tag, seed))


def identity(variant):
    params = variant['algo_params']
    if not mixed(params):
        return None
    _, _, provenance, _ = bundle(params)
    return dict(environment=variant['env_name'], dataset_tag=params.get('dataset_tag', 'mixed_rme_v3'),
                dynamics_max_epochs=variant.get('dynamics_max_epochs'), dataset_sha256=file_hash(Path(params['train_artifacts_dir']) / 'dataset_manifest.json'),
                task_manifest_sha256=file_hash(Path(params['train_artifacts_dir']) / 'task_manifest.json'),
                eval_dataset_sha256=provenance['old_dataset_manifest_sha256'],
                normalization='GENTLE RunningMeanStd, mixed training current observations only',
                use_next_obs_in_context=params['use_next_obs_in_context'])


def write_identity(directory, variant, normalizer=None):
    value = identity(variant)
    if value is None:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'dataset_identity.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    if normalizer is not None:
        np.savez(directory / 'obs_normalizer.npz', mean=normalizer.mean, var=normalizer.var, count=normalizer.count)


def check_identity(directory, variant):
    expected = identity(variant)
    if expected is not None:
        require(read_json(Path(directory) / 'dataset_identity.json') == expected,
                'Pretrained weights belong to a different dataset/config: ' + str(directory))


def prepare_environment(env, variant):
    params = variant['algo_params']
    if not mixed(params):
        return
    dataset, manifest, provenance, _ = bundle(params)
    require(manifest['environment'] == variant['env_name'], 'Configured environment mismatch')
    require(manifest['train_tasks'] == list(env.get_all_task_idx())[:variant['n_train_tasks']], 'Train split mismatch')
    require(manifest['eval_tasks'] == list(env.get_all_task_idx())[-variant['n_eval_tasks']:], 'Eval split mismatch')
    require(manifest['horizon'] == params['max_path_length'] == env._max_episode_steps, 'Horizon mismatch')
    raw = env._wrapped_env
    require('non_mujoco' not in type(raw).__module__ or variant['env_name'] == 'point-robot',
            'Mixed experiment requires real MuJoCo, not placeholder environments')
    dimensions = dataset['files'][0]['dimensions']
    require(dimensions['observation'] == int(np.prod(env.observation_space.shape)) and
            dimensions['action'] == int(np.prod(env.action_space.shape)), 'Environment dimensions mismatch')
    require(np.array_equal(raw.action_space.low, provenance['action_low']) and
            np.array_equal(raw.action_space.high, provenance['action_high']), 'Physical action bounds mismatch')
    actual = raw.goals if variant['env_name'] == 'point-robot' else raw.tasks
    def compare(a, b):
        if isinstance(b, dict):
            return isinstance(a, dict) and set(a) == set(b) and all(compare(a[k], b[k]) for k in b)
        if isinstance(b, list) and b and isinstance(b[0], dict):
            return len(a) == len(b) and all(compare(x, y) for x, y in zip(a, b))
        left, right = np.asarray(a), np.asarray(b)
        return left.shape == right.shape and np.allclose(left, right, rtol=1e-10, atol=1e-12)
    require(compare(actual, manifest['tasks']), 'Environment task parameters differ from the dataset')
    if variant['env_name'] == 'point-robot':
        raw.goals = np.asarray(manifest['tasks'])
    else:
        raw.tasks = [{key: np.asarray(value) if isinstance(value, list) else value
                      for key, value in task.items()} for task in manifest['tasks']]
    raw.reset_task(0)


def check_normalizer(directory, variant, normalizer):
    if not mixed(variant['algo_params']):
        return
    with np.load(Path(directory) / 'obs_normalizer.npz', allow_pickle=False) as saved:
        for name in ('mean', 'var'):
            actual = getattr(normalizer, name)
            require(saved[name].shape == actual.shape and np.allclose(saved[name], actual, rtol=1e-10, atol=1e-12),
                    'Pretrained observation normalization mismatch: ' + str(directory))
