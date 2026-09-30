"""Deterministic, paired offline data for checkpoint diagnostics.

This module intentionally depends only on NumPy and the standard library. Raw
observations are preserved here; callers apply each run's observation statistics
after selecting a shared bank. Existing trajectory files are trusted NumPy
object arrays, as in the training data loader.
"""

import hashlib
import io
import numbers
from pathlib import Path
import re

import numpy as np


TRANSITION_FIELDS = (
    'observations', 'actions', 'rewards', 'next_observations', 'terminals',
)


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Real)
            or not np.isfinite(value) or int(value) != value
            or value < minimum):
        raise ValueError('{} must be an integer >= {}'.format(name, minimum))
    return int(value)


def _task_ids(task_ids):
    result = [_integer(task, 'task_id') for task in task_ids]
    if not result or len(set(result)) != len(result):
        raise ValueError('task_ids must be nonempty and contain no duplicates')
    return sorted(result)


def _numeric_finite(array, label):
    array = np.asarray(array)
    if (not np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.complexfloating)):
        raise ValueError('{} must contain real numeric values'.format(label))
    if not np.isfinite(array).all():
        raise ValueError('{} contains nonfinite values'.format(label))
    return array


def _trajectory_column(rows, column, width, label, allow_scalar=False):
    values = []
    for row_number, row in enumerate(rows):
        value = np.asarray(row[column])
        if allow_scalar and value.ndim == 0:
            value = value.reshape(1)
        if value.shape != (width,):
            raise ValueError('{} row {} has shape {}; expected ({},)'.format(
                label, row_number, value.shape, width))
        values.append(_numeric_finite(value, label))
    return np.stack(values)


def load_offline_pools(data_dir, task_ids, n_trj, epoch, obs_dim, action_dim):
    """Read exactly the requested trajectory indices and tasks.

    Returns ``(pools, manifest)``. Manifest entries are ordered by trajectory
    index, task ID, then numeric epoch and filename. Within each task, row IDs
    enumerate that same file order. Every requested (task, trajectory index)
    must have a file; ``epoch=None`` admits all numeric steps for that pair.
    """
    task_ids = _task_ids(task_ids)
    n_trj = _integer(n_trj, 'n_trj', 1)
    obs_dim = _integer(obs_dim, 'obs_dim', 1)
    action_dim = _integer(action_dim, 'action_dim', 1)
    if epoch is not None:
        epoch = _integer(epoch, 'epoch')
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError('Offline data directory does not exist: {}'.format(data_dir))

    chunks = {task: {field: [] for field in TRANSITION_FIELDS} for task in task_ids}
    row_counts = {task: 0 for task in task_ids}
    manifest = []
    for trajectory_index in range(n_trj):
        for task in task_ids:
            directory = data_dir / 'goal_idx{}'.format(task)
            pattern = 'trj_evalsample{}_step{}.npy'.format(
                trajectory_index, '*' if epoch is None else epoch)
            filename_regex = re.compile(
                r'trj_evalsample{}_step(\d+)\.npy'.format(trajectory_index))
            matches = []
            for path in directory.glob(pattern):
                match = filename_regex.fullmatch(path.name)
                if match is not None and path.is_file():
                    matches.append((int(match.group(1)), path))
            matches.sort(key=lambda item: (item[0], item[1].name))
            if not matches:
                raise FileNotFoundError(
                    'Missing task {} trajectory {}: {}'.format(task, trajectory_index, directory / pattern))
            for file_epoch, path in matches:
                payload = path.read_bytes()
                try:
                    rows = np.load(io.BytesIO(payload), allow_pickle=True)
                except Exception as error:
                    raise ValueError('Unable to load trajectory {}'.format(path)) from error
                if (not isinstance(rows, np.ndarray) or rows.ndim != 2
                        or rows.shape[1] != 4 or rows.shape[0] == 0):
                    raise ValueError('{} must contain a nonempty [N, 4] trajectory'.format(path))
                parsed = {}
                for field, column, width in (
                        ('observations', 0, obs_dim), ('actions', 1, action_dim),
                        ('rewards', 2, 1), ('next_observations', 3, obs_dim)):
                    parsed[field] = _trajectory_column(
                        rows, column, width, '{} {}'.format(path, field),
                        allow_scalar=field == 'rewards')
                parsed['terminals'] = np.zeros((len(rows), 1), dtype=np.uint8)
                parsed['terminals'][-1, 0] = 1
                for field in TRANSITION_FIELDS:
                    chunks[task][field].append(parsed[field])
                first_row = row_counts[task]
                row_counts[task] += len(rows)
                manifest.append({
                    'path': str(path.resolve()),
                    'sha256': hashlib.sha256(payload).hexdigest(),
                    'row_count': len(rows), 'task_id': task,
                    'trajectory_index': trajectory_index, 'epoch': file_epoch,
                    'row_id_start': first_row, 'row_id_stop': row_counts[task],
                })
    pools = {}
    for task in task_ids:
        pools[task] = {field: np.concatenate(chunks[task][field], axis=0)
                       for field in TRANSITION_FIELDS}
        pools[task]['row_ids'] = np.arange(row_counts[task], dtype=np.int64)
    return pools, manifest


def compute_observation_stats(pools):
    """Reproduce one RunningMeanStd.update over the supplied pools' raw obs.

    Call with training pools only. In particular, neither next observations nor
    evaluation observations contribute. NumPy's input dtype reduction semantics
    match the training RunningMeanStd implementation.
    """
    task_ids = _task_ids(pools)
    observations = []
    width = None
    for task in task_ids:
        array = _numeric_finite(pools[task]['observations'], 'observations')
        if array.ndim != 2 or not len(array) or array.shape[1] == 0:
            raise ValueError('observations must be a nonempty [N, obs_dim] array')
        if width is not None and array.shape[1] != width:
            raise ValueError('Observation widths differ between tasks')
        width = array.shape[1]
        observations.append(array)
    observations = np.concatenate(observations, axis=0)
    batch_mean = np.mean(observations, axis=0)
    batch_var = np.var(observations, axis=0)
    batch_count = len(observations)
    initial_mean = np.zeros(width, dtype=np.float64)
    initial_var = np.ones(width, dtype=np.float64)
    initial_count = 1e-4
    delta = batch_mean - initial_mean
    count = initial_count + batch_count
    mean = initial_mean + delta * batch_count / count
    m_a = initial_var * initial_count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + np.square(delta) * initial_count * batch_count / count
    var = m_2 / count
    if not np.isfinite(mean).all() or not np.isfinite(var).all():
        raise ValueError('Observation moments overflowed; check data magnitude and dtype')
    return {'mean': mean, 'var': var, 'count': float(count)}


def _validate_pool(pool, task):
    arrays = {}
    count = None
    for field in TRANSITION_FIELDS:
        if field not in pool:
            raise ValueError('Task {} is missing {}'.format(task, field))
        array = _numeric_finite(pool[field], 'task {} {}'.format(task, field))
        if array.ndim != 2 or not len(array) or array.shape[1] == 0:
            raise ValueError('Task {} {} must have shape [N, D]'.format(task, field))
        if count is not None and len(array) != count:
            raise ValueError('Task {} has unpaired transition fields'.format(task))
        count = len(array)
        arrays[field] = array
    if arrays['observations'].shape != arrays['next_observations'].shape:
        raise ValueError('Task {} has incompatible observation shapes'.format(task))
    if arrays['rewards'].shape[1] != 1 or arrays['terminals'].shape[1] != 1:
        raise ValueError('Rewards and terminals must have shape [N, 1]')
    if not np.isin(arrays['terminals'], [0, 1]).all():
        raise ValueError('Terminals must be zero or one')
    row_ids = np.asarray(pool.get('row_ids'))
    if (row_ids.shape != (count,) or not np.issubdtype(row_ids.dtype, np.integer)
            or (row_ids < 0).any() or len(np.unique(row_ids)) != count):
        raise ValueError('Task {} needs unique nonnegative integer row_ids'.format(task))
    return arrays, row_ids


def build_offline_bank(pools, context_size, repeats, probe_size, seed, capacity=None):
    """Select paired, disjoint contexts and probes without global RNG effects.

    All context repetitions and probes are mutually disjoint within each task.
    When supplied, capacity restricts sampling to the final capacity rows, the
    set retained by a replay ring after sequential insertion. Row IDs always
    refer to the original full task pool, even after this restriction.
    """
    task_ids = _task_ids(pools)
    context_size = _integer(context_size, 'context_size', 1)
    repeats = _integer(repeats, 'repeats', 1)
    probe_size = _integer(probe_size, 'probe_size', 1)
    seed = _integer(seed, 'seed')
    if capacity is not None:
        capacity = _integer(capacity, 'capacity', 1)
    rng = np.random.default_rng(seed)
    needed = context_size * repeats + probe_size
    bank = {}
    for task in task_ids:
        arrays, row_ids = _validate_pool(pools[task], task)
        offset = 0 if capacity is None else max(0, len(row_ids) - capacity)
        available = len(row_ids) - offset
        if available < needed:
            raise ValueError('Task {} has {} available rows but {} are required for '
                             '{} disjoint contexts of {} rows and {} probes'.format(
                                 task, available, needed, repeats, context_size, probe_size))
        selected = rng.permutation(available)[:needed] + offset
        context_indices = selected[:context_size * repeats].reshape(repeats, context_size)
        probe_indices = selected[context_size * repeats:]
        bank[task] = {
            'contexts': {field: array[context_indices].copy() for field, array in arrays.items()},
            'context_row_ids': row_ids[context_indices].copy(),
            'probes': {field: array[probe_indices].copy() for field, array in arrays.items()},
            'probe_row_ids': row_ids[probe_indices].copy(),
        }
    return bank
