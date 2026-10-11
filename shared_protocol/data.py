# Shared artifact schema ported from the user's FOCAL/focal_shared (2026-10-11).
"""Strict four-field trajectory loading, train-only statistics and fingerprints."""
from pathlib import Path
import shutil

import numpy as np

from shared_protocol.common import PROTOCOL, digest, file_hash, read_json, write_json

FIELDS = ('observations', 'actions', 'rewards', 'next_observations', 'terminals')


def read_trajectory(path, obs_dim, action_dim, horizon):
    # Legacy files are trusted local NumPy pickles; never download/load untrusted files.
    raw = np.load(path, allow_pickle=True)
    if raw.ndim != 2 or raw.shape[1] != 4 or not 0 < len(raw) <= horizon:
        raise ValueError(f'{path}: expected 1..{horizon} rows of [s,a,r,s_next], got {raw.shape}')
    values = []
    for col, dim in enumerate((obs_dim, action_dim, 1, obs_dim)):
        rows = [np.asarray(row) for row in raw[:, col]]
        if col == 2:
            rows = [x.reshape(1) if x.ndim == 0 else x for x in rows]
        if any(x.shape != (dim,) for x in rows):
            raise ValueError(f'{path}: field {col} must have shape ({dim},)')
        array = np.stack(rows)
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise ValueError(f'{path}: field {col} must contain real numeric values')
        if not np.isfinite(array).all():
            raise ValueError(f'{path}: non-finite field {col}')
        values.append(array)
    terminal = np.zeros((len(raw), 1), dtype=np.float64)
    terminal[-1] = 1
    return dict(zip(FIELDS, values + [terminal]))


def load_raw(cfg, obs_dim, action_dim, action_coordinates):
    if action_coordinates not in ('normalized', 'physical'):
        raise ValueError('Explicit data action coordinates required')
    root = Path(cfg['data_dir'])
    entries, pools = [], {}
    train, test = cfg['train_tasks'], cfg['eval_tasks']
    expected = [(task, n, root / f'goal_idx{task}' /
                 f'trj_evalsample{n}_step{cfg["data_checkpoint"]}.npy')
                for task in sorted(set(train + test)) for n in range(cfg['n_trj'])]
    missing = [str(path) for _, _, path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} required trajectories missing; first: {missing[:5]}')
    for task, n, path in expected:
        batch = read_trajectory(path, obs_dim, action_dim, cfg['horizon'])
        pools.setdefault(task, []).append(batch)
        entries.append(dict(task_id=task, splits=[name for name, ids in [('train', train), ('test', test)] if task in ids],
                            path=path.relative_to(root).as_posix(), sha256=file_hash(path),
                            rows=len(batch['observations']), checkpoint=cfg['data_checkpoint'],
                            dimensions=dict(observation=obs_dim, action=action_dim, reward=1, next_observation=obs_dim)))
    pools = {task: {key: np.concatenate([b[key] for b in batches]) for key in FIELDS}
             for task, batches in pools.items()}
    manifest = dict(protocol_id=PROTOCOL, environment=cfg['environment'], files=entries,
                    task_rows={str(t): len(p['observations']) for t, p in pools.items()},
                    terminal_mode='legacy_episode_end', observation_coordinates='raw',
                    data_action_coordinates=action_coordinates, policy_action_coordinates='normalized',
                    reward_coordinates='raw')
    return pools, manifest


def fit_normalizer(pools, train_tasks):
    from rlkit.data_management.fixed_context_data import compute_observation_stats
    result = compute_observation_stats({t: pools[t] for t in train_tasks})
    result['count'] = np.asarray(result['count'])
    return result


def load_normalizer(path, expected=None):
    with np.load(path, allow_pickle=False) as src:
        values = {key: src[key] for key in ('mean', 'var', 'count')}
        if any(not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x) for x in values.values()):
            raise ValueError('Normalizer must contain real numeric arrays')
        result = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
    if (result['mean'].ndim != 1 or result['var'].shape != result['mean'].shape or
            result['count'].shape != () or result['count'] <= 0 or
            any(not np.isfinite(x).all() for x in result.values()) or (result['var'] < 0).any()):
        raise ValueError('Invalid observation normalizer')
    if expected is not None:
        for key in result:
            if result[key].shape != expected[key].shape or not np.allclose(result[key], expected[key], rtol=1e-5, atol=1e-6):
                raise ValueError(f'Normalizer {key} does not match selected training current observations')
    return result


def normalize(obs, stats):
    return (obs - stats['mean']) / np.sqrt(stats['var'] + 1e-8)


def transform_pools(pools, stats, action_coordinates, low, high):
    transformed = {}
    for task, pool in pools.items():
        out = {key: value.copy() for key, value in pool.items()}
        for key in ('observations', 'next_observations'):
            out[key] = normalize(out[key], stats)
        if action_coordinates == 'physical':
            out['actions'] = 2 * (out['actions'] - low) / (high - low) - 1
        if np.max(np.abs(out['actions'])) > 1 + 1e-6:
            raise ValueError(f'Task {task}: actions outside declared coordinates; refusing to clip data')
        transformed[task] = {key: value.astype(np.float32) for key, value in out.items()}
        if any(not np.isfinite(value).all() for value in transformed[task].values()):
            raise ValueError(f'Task {task}: non-finite values after float32 conversion/normalization')
    return transformed


def prepare_dataset(cfg, env, out, action_coordinates, shared_normalizer=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pools, manifest = load_raw(cfg, env.observation_space.shape[0], env.action_space.shape[0], action_coordinates)
    expected = fit_normalizer(pools, cfg['train_tasks'])
    stats = load_normalizer(shared_normalizer, expected) if shared_normalizer else expected
    normalizer_path = out / 'observation_normalizer.npz'
    if shared_normalizer:
        if Path(shared_normalizer).resolve() != normalizer_path.resolve():
            shutil.copyfile(shared_normalizer, normalizer_path)
    else:
        np.savez(normalizer_path, **stats)
    manifest['normalizer_sha256'] = file_hash(normalizer_path)
    manifest['task_manifest_sha256'] = file_hash(out / 'task_manifest.json')
    transform_pools(pools, stats, action_coordinates, env.action_space.low, env.action_space.high)
    write_json(out / 'dataset_manifest.json', manifest)
    return manifest


def load_prepared(cfg, env, artifacts):
    artifacts = Path(artifacts)
    saved = read_json(artifacts / 'dataset_manifest.json')
    raw, actual = load_raw(cfg, env.observation_space.shape[0], env.action_space.shape[0], saved['data_action_coordinates'])
    actual['normalizer_sha256'] = file_hash(artifacts / 'observation_normalizer.npz')
    actual['task_manifest_sha256'] = file_hash(artifacts / 'task_manifest.json')
    if digest(actual) != digest(saved):
        raise ValueError('Dataset/artifact fingerprint mismatch; prepare all methods against the same artifacts')
    stats = load_normalizer(artifacts / 'observation_normalizer.npz', fit_normalizer(raw, cfg['train_tasks']))
    return transform_pools(raw, stats, saved['data_action_coordinates'], env.action_space.low, env.action_space.high), stats
