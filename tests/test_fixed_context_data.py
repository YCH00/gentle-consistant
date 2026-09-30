"""Fixed diagnostics banks retain pair identity and training normalization."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from rlkit.data_management.fixed_context_data import (
    TRANSITION_FIELDS,
    build_offline_bank,
    compute_observation_stats,
    load_offline_pools,
)
from rlkit.torch.pytorch_util import RunningMeanStd


def save_trajectory(root, task=0, trajectory=0, epoch=10, count=5, base=0, dtype=np.float64):
    directory = root / 'goal_idx{}'.format(task)
    directory.mkdir(parents=True, exist_ok=True)
    rows = np.empty((count, 4), dtype=object)
    for i in range(count):
        value = base + i
        rows[i, 0] = np.asarray([value, value + 1], dtype=dtype)
        rows[i, 1] = np.asarray([value + 2], dtype=dtype)
        rows[i, 2] = dtype(value + 3)
        rows[i, 3] = np.asarray([value + 4, value + 5], dtype=dtype)
    path = directory / 'trj_evalsample{}_step{}.npy'.format(trajectory, epoch)
    np.save(path, rows)
    return path


def load(root, tasks=(0,), n_trj=1, epoch=10):
    return load_offline_pools(root, tasks, n_trj, epoch, obs_dim=2, action_dim=1)


def pool(count=60, base=0):
    value = np.arange(count, dtype=np.float64)[:, None] + base
    return {
        'observations': np.concatenate([value, value + 1], axis=1),
        'actions': value + 2, 'rewards': value + 3,
        'next_observations': np.concatenate([value + 4, value + 5], axis=1),
        'terminals': (value % 5 == 4).astype(np.uint8),
        'row_ids': np.arange(count, dtype=np.int64),
    }


def assert_paired(fields):
    value = fields['observations'][..., :1]
    np.testing.assert_array_equal(fields['actions'], value + 2)
    np.testing.assert_array_equal(fields['rewards'], value + 3)
    np.testing.assert_array_equal(fields['next_observations'][..., :1], value + 4)
    np.testing.assert_array_equal(fields['next_observations'][..., 1:], value + 5)


def test_loader_preserves_pairing_requested_files_and_manifest(tmp_path):
    expected_paths = []
    for trajectory in range(2):
        for task in (0, 2):
            expected_paths.append(save_trajectory(tmp_path, task, trajectory, base=100 * task + 10 * trajectory))
    save_trajectory(tmp_path, task=1, base=999)
    save_trajectory(tmp_path, task=0, trajectory=2, base=999)
    save_trajectory(tmp_path, task=0, epoch=9, base=999)
    pools, manifest = load(tmp_path, tasks=(2, 0), n_trj=2)
    assert list(pools) == [0, 2]
    assert [entry['path'] for entry in manifest] == [str(path.resolve()) for path in expected_paths]
    for entry in manifest:
        assert entry['sha256'] == hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()
        assert entry['row_count'] == 5
        assert entry['row_id_stop'] - entry['row_id_start'] == 5
        assert entry['epoch'] == 10
    for task, data in pools.items():
        assert_paired(data)
        np.testing.assert_array_equal(data['row_ids'], np.arange(10))
        np.testing.assert_array_equal(data['terminals'].ravel(), [0, 0, 0, 0, 1] * 2)
        np.testing.assert_array_equal(data['observations'][:5, 0], np.arange(5) + 100 * task)


def test_none_epoch_matches_all_numeric_steps_in_fixed_order(tmp_path):
    for epoch in (100, 2, 20):
        save_trajectory(tmp_path, epoch=epoch, base=epoch)
    save_trajectory(tmp_path, trajectory=1, epoch=2, base=999)
    (tmp_path / 'goal_idx0' / 'trj_evalsample0_stepbad.npy').write_bytes(b'not selected')
    pools, manifest = load(tmp_path, epoch=None)
    assert [entry['epoch'] for entry in manifest] == [2, 20, 100]
    np.testing.assert_array_equal(pools[0]['observations'][::5, 0], [2, 20, 100])


@pytest.mark.parametrize('tasks,n_trj,epoch', [((0, 1), 1, 10), ((0,), 2, 10), ((0,), 1, 11), ((0,), 2, None)])
def test_loader_requires_each_task_and_trajectory(tmp_path, tasks, n_trj, epoch):
    save_trajectory(tmp_path)
    with pytest.raises(FileNotFoundError, match='Missing task'):
        load(tmp_path, tasks, n_trj, epoch)


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('column', [0, 1, 2, 3])
def test_loader_rejects_nonfinite_fields(tmp_path, bad_value, column):
    path = save_trajectory(tmp_path)
    rows = np.load(path, allow_pickle=True)
    if column == 2:
        rows[1, column] = bad_value
    else:
        rows[1, column][0] = bad_value
    np.save(path, rows)
    with pytest.raises(ValueError, match='nonfinite'):
        load(tmp_path)


@pytest.mark.parametrize('kind', ['empty', 'columns', 'obs_width', 'action_width', 'reward_width', 'complex', 'text', 'broken'])
def test_loader_rejects_malformed_trajectories(tmp_path, kind):
    path = save_trajectory(tmp_path)
    rows = np.load(path, allow_pickle=True)
    if kind == 'empty':
        rows = rows[:0]
    elif kind == 'columns':
        rows = rows[:, :3]
    elif kind == 'obs_width':
        rows[0, 0] = np.zeros(3)
    elif kind == 'action_width':
        rows[0, 1] = np.zeros(2)
    elif kind == 'reward_width':
        rows[0, 2] = np.zeros(2)
    elif kind == 'complex':
        rows[0, 2] = 1 + 2j
    elif kind == 'text':
        rows[0, 2] = '3'
    elif kind == 'broken':
        path.write_bytes(b'broken')
        with pytest.raises(ValueError, match='Unable to load'):
            load(tmp_path)
        return
    np.save(path, rows)
    with pytest.raises(ValueError):
        load(tmp_path)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64])
def test_stats_equal_real_running_mean_std_and_exclude_next_obs(dtype):
    pools = {0: pool(11), 3: pool(9, base=20)}
    for data in pools.values():
        data['observations'] = data['observations'].astype(dtype)
        data['next_observations'].fill(1e9)
    observations = np.concatenate([pools[0]['observations'], pools[3]['observations']])
    expected = RunningMeanStd(shape=2)
    expected.update(observations)
    actual = compute_observation_stats(pools)
    np.testing.assert_array_equal(actual['mean'], expected.mean)
    np.testing.assert_array_equal(actual['var'], expected.var)
    assert actual['count'] == expected.count


def test_bank_has_disjoint_paired_contexts_and_probes_and_is_repeatable():
    pools = {2: pool(), 0: pool(base=100)}
    first = build_offline_bank(pools, 7, 3, 13, seed=14)
    second = build_offline_bank(dict(reversed(list(pools.items()))), 7, 3, 13, seed=14)
    third = build_offline_bank(pools, 7, 3, 13, seed=15)
    assert list(first) == [0, 2]
    for task in first:
        bank = first[task]
        assert bank['contexts']['observations'].shape == (3, 7, 2)
        assert bank['probes']['observations'].shape == (13, 2)
        selected = np.concatenate([bank['context_row_ids'].ravel(), bank['probe_row_ids']])
        assert len(np.unique(selected)) == 34
        assert_paired(bank['contexts'])
        assert_paired(bank['probes'])
        np.testing.assert_array_equal(bank['context_row_ids'], second[task]['context_row_ids'])
        np.testing.assert_array_equal(bank['probe_row_ids'], second[task]['probe_row_ids'])
        assert not np.array_equal(bank['context_row_ids'], third[task]['context_row_ids'])
        for field in TRANSITION_FIELDS:
            np.testing.assert_array_equal(bank['contexts'][field], pools[task][field][bank['context_row_ids']])
            np.testing.assert_array_equal(bank['probes'][field], pools[task][field][bank['probe_row_ids']])
    first[0]['contexts']['observations'].fill(9999)
    assert pools[0]['observations'].max() < 9999


def test_bank_does_not_advance_global_numpy_rng():
    np.random.seed(987)
    original = np.random.get_state()
    build_offline_bank({0: pool()}, 7, 3, 13, seed=14)
    after = np.random.get_state()
    assert original[0] == after[0]
    np.testing.assert_array_equal(original[1], after[1])
    assert original[2:] == after[2:]


def test_capacity_keeps_last_rows_and_original_identifiers():
    bank = build_offline_bank({0: pool(60)}, 5, 2, 10, seed=14, capacity=20)[0]
    selected = np.concatenate([bank['context_row_ids'].ravel(), bank['probe_row_ids']])
    np.testing.assert_array_equal(np.sort(selected), np.arange(40, 60))
    assert_paired(bank['contexts'])
    assert_paired(bank['probes'])


@pytest.mark.parametrize('capacity', [None, 19])
def test_too_few_unique_rows_is_an_error(capacity):
    with pytest.raises(ValueError, match='required'):
        build_offline_bank({0: pool(20)}, 5, 2, 11, seed=14, capacity=capacity)


@pytest.mark.parametrize('field', TRANSITION_FIELDS)
def test_bank_rejects_unpaired_arrays(field):
    data = pool()
    data[field] = data[field][:-1]
    with pytest.raises(ValueError, match='unpaired'):
        build_offline_bank({0: data}, 2, 2, 2, seed=0)


def test_bank_rejects_duplicate_row_ids():
    data = pool()
    data['row_ids'][1] = data['row_ids'][0]
    with pytest.raises(ValueError, match='row_ids'):
        build_offline_bank({0: data}, 2, 2, 2, seed=0)


@pytest.mark.parametrize('name,value', [('context_size', 0), ('repeats', 1.5), ('probe_size', -1), ('seed', True), ('capacity', 0)])
def test_bank_validates_sizes(name, value):
    kwargs = dict(context_size=2, repeats=2, probe_size=2, seed=0, capacity=None)
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        build_offline_bank({0: pool()}, **kwargs)


@pytest.mark.parametrize('tasks', [(), (0, 0), (-1,), (True,)])
def test_loader_validates_task_ids(tmp_path, tasks):
    with pytest.raises(ValueError):
        load(tmp_path, tasks)
