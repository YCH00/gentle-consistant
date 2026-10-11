"""Shared schemas and real Point backend; synthetic trajectories live only in tmp_path."""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from shared_protocol.common import digest, file_hash, read_json, write_json
from shared_protocol.data import fit_normalizer, load_raw, load_normalizer
from shared_protocol.environments import make_env, verify_reference, export_or_check_tasks
from shared_protocol.export import export_environment
from shared_protocol.runtime import (ROOT, CORE_FILES, configure, fill_buffer, install_normalizer,
    load_bundle, make_training_env, persist_bundle, protocol_config, pretrained_dir,
    check_pretrained, bind_evaluation_run)


def write_trajectory(path, task, trajectory, count=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.empty((count, 4), dtype=object)
    for i in range(count):
        obs = np.asarray([task / 10 + i / 50, trajectory / 100 + i / 50], dtype=np.float32)
        rows[i] = [obs, np.array([.25, -.25], dtype=np.float32), float(-task-i), obs + 1000]
    np.save(path, rows)


@pytest.fixture(scope='module')
def published(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic_shared_TEST_ONLY')
    data = root / 'data'
    cfg = protocol_config('point-robot', data)
    for t in range(20):
        for n in range(100):
            write_trajectory(data / f'goal_idx{t}' / f'trj_evalsample{n}_step100000.npy', t, n)
    target, status = export_environment('point-robot', data, root / 'test_exports',
                                        'normalized', 'SYNTHETIC TEST FIXTURE, not experimental data')
    assert status == 'exported'
    return cfg, target


def test_source_pin_and_six_config_contract():
    root, fingerprint = verify_reference(ROOT)
    assert root == ROOT and len(fingerprint) == 64
    from configs.default import default_config
    for name in ('point-robot', 'ant-dir', 'cheetah-vel', 'cheetah-dir', 'hopper-rand-params', 'walker-rand-params'):
        cfg = protocol_config(name, '.')
        overrides = read_json(ROOT / 'configs' / (name + '.json'))
        algo = {**default_config['algo_params'], **overrides['algo_params']}
        assert cfg['n_trj'] == algo['n_trj']
        assert cfg['horizon'] == algo['max_path_length']
        assert cfg['data_checkpoint'] == algo['train_epoch'] == algo['eval_epoch']


def test_export_statistics_and_reuse(published):
    cfg, target = published
    env = make_env(cfg, ROOT)
    bundle = load_bundle(cfg, env, target)
    expected_goals = np.random.RandomState(1337).uniform(-1, 1, (20, 2))
    np.testing.assert_array_equal(env.goals, expected_goals)
    assert env._goal == env.goals[0]
    assert len(bundle['manifest']['files']) == 2000
    assert bundle['stats']['count'] == 2000 + 1e-4  # 10 tasks x 100 files x 2 rows
    obs = np.concatenate([bundle['pools'][t]['observations'] for t in cfg['train_tasks']])
    from rlkit.torch.pytorch_util import RunningMeanStd
    reference = RunningMeanStd(shape=2)
    reference.update(obs)
    np.testing.assert_array_equal(reference.mean, bundle['stats']['mean'])
    np.testing.assert_array_equal(reference.var, bundle['stats']['var'])
    old_hashes = bundle['hashes']
    _, status = export_environment('point-robot', cfg['data_dir'], target.parent, 'normalized', 'test reuse')
    assert status == 'verified_and_reused'
    assert old_hashes == {f: file_hash(target / f) for f in CORE_FILES}
    with np.load(target / 'observation_normalizer.npz', allow_pickle=False) as archive:
        assert archive['count'].shape == ()
        assert archive['format_version'].item() == 1
    env.close()


def test_train_only_statistics_are_independent_of_next_and_test(published):
    cfg, target = published
    raw, _ = load_raw(cfg, 2, 2, 'normalized')
    a = fit_normalizer(raw, cfg['train_tasks'])
    for t, p in raw.items():
        p['next_observations'][:] = -1e5
        if t not in cfg['train_tasks']:
            p['observations'][:] = 1e6
    b = fit_normalizer(raw, cfg['train_tasks'])
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize('filename', CORE_FILES)
def test_artifact_tampering_rejected(published, tmp_path, filename):
    cfg, source = published
    for f in CORE_FILES:
        shutil.copyfile(source / f, tmp_path / f)
    path = tmp_path / filename
    if filename == 'observation_normalizer.npz':
        np.savez(path, mean=[100, 100], var=[1, 1], count=2000.0001)
    else:
        value = read_json(path)
        if filename == 'task_manifest.json':
            value['tasks'][0][0] += .01
        else:
            value['task_rows']['0'] += 1
        write_json(path, value)
    env = make_env(cfg, ROOT)
    with pytest.raises(ValueError):
        load_bundle(cfg, env, tmp_path)
    env.close()


def test_missing_and_modified_raw_data_rejected(published):
    cfg, target = published
    path = Path(cfg['data_dir']) / 'goal_idx19/trj_evalsample99_step100000.npy'
    original = path.read_bytes()
    try:
        path.unlink()
        with pytest.raises(FileNotFoundError, match='required trajectories missing'):
            load_raw(cfg, 2, 2, 'normalized')
        write_trajectory(path, 19, 98)
        env = make_env(cfg, ROOT)
        with pytest.raises(ValueError, match='fingerprint'):
            load_bundle(cfg, env, target)
        env.close()
    finally:
        path.write_bytes(original)


def variant_for(cfg, artifacts):
    from configs.default import default_config
    variant = copy.deepcopy(default_config)
    override = read_json(ROOT / 'configs/point-robot.json')
    for key, value in override.items():
        if isinstance(value, dict):
            variant[key].update(value)
        else:
            variant[key] = value
    configure(variant, artifacts, cfg['data_dir'])
    return variant


def test_runtime_uses_exact_bundle_without_refit(published, tmp_path):
    cfg, target = published
    variant = variant_for(cfg, target)
    env = make_training_env(variant)
    bundle = env.shared_bundle
    class Normalizer:
        def update(self, *args):
            pytest.fail('Shared statistics must never update')
    normalizer = Normalizer()
    install_normalizer(normalizer, bundle)
    np.testing.assert_array_equal(normalizer.mean, bundle['stats']['mean'])
    captured = []
    buffer = SimpleNamespace(add_sample=lambda *a, **k: captured.append(a))
    fill_buffer(buffer, [0], bundle)
    assert len(captured) == 200
    assert captured[0][4].item() == 0 and captured[1][4].item() == 1
    first_raw = bundle['pools'][0]['observations'][0]
    expected = (first_raw - normalizer.mean) / np.sqrt(normalizer.var + 1e-8)
    np.testing.assert_array_equal(captured[0][1], expected)
    persist_bundle(bundle, tmp_path)
    assert bundle['hashes'] == {f: file_hash(tmp_path / f) for f in CORE_FILES}
    moved = copy.deepcopy(variant)
    bind_evaluation_run(moved, tmp_path)
    assert Path(moved['shared_artifacts']) == tmp_path
    with pytest.raises(ValueError, match='legacy'):
        bind_evaluation_run({}, tmp_path, target)
    assert 'shared-' in str(pretrained_dir('dynamics', variant, 0))
    check_pretrained(tmp_path, variant)
    env.close()


def test_training_and_encoder_pretraining_buffer_paths(published, monkeypatch):
    cfg, target = published
    variant = variant_for(cfg, target)
    env = make_training_env(variant)
    import pretrain_encoder_decoder as pretrain
    def no_update(*args):
        pytest.fail('Refit on shared path')
    monkeypatch.setattr(pretrain.ptu.RunningMeanStd, 'update', no_update)
    variant['algo_params']['replay_buffer_size'] = 250
    buffer, normalizer = pretrain.build_train_buffer(variant, env, cfg['train_tasks'], 2)
    assert buffer.num_steps_can_sample(0) == 200
    # Execute the actual init_buffer method without importing plotting/timer dependencies.
    tree = ast.parse((ROOT / 'rlkit/core/rl_algorithm.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OfflineMetaRLAlgorithm')
    func = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'init_buffer')
    namespace = {}
    exec(compile(ast.Module(body=[func], type_ignores=[]), 'rl_algorithm.py', 'exec'), namespace)
    captured = []
    target_buffer = SimpleNamespace(add_sample=lambda *a, **k: captured.append(a))
    algorithm = SimpleNamespace(env=env, train_tasks=cfg['train_tasks'], eval_tasks=cfg['eval_tasks'],
        obs_normalizer=normalizer, train_buffer=target_buffer, eval_buffer=target_buffer, load_eval_buffer=True)
    namespace['init_buffer'](algorithm, {})
    assert len(captured) == 4000
    env.close()


def test_failed_export_never_publishes(published, tmp_path):
    cfg, target = published
    bogus = tmp_path / 'bad.npz'
    np.savez(bogus, mean=[0, 0], var=[1, 1], count=1)
    with pytest.raises(ValueError, match='Normalizer'):
        export_environment('point-robot', cfg['data_dir'], tmp_path / 'out', 'normalized', 'test', bogus)
    assert not (tmp_path / 'out/point-robot').exists()


def test_dynamics_entry_consumes_shared_observations(published, tmp_path, monkeypatch):
    import pretrain_dynamics as entry
    import shared_protocol.runtime as runtime
    cfg, target = published
    variant = variant_for(cfg, target)
    variant['util_params']['use_gpu'] = False
    variant['algo_params']['replay_buffer_size'] = 250
    observed = {}
    class RecordingDynamics:
        def __init__(self, **kwargs):
            pass
        def to(self, **kwargs):
            return self
        def set_task_idx(self, task):
            self.task = task
        def train(self, data):
            observed[self.task] = data
        def save(self, path):
            pass
    monkeypatch.setattr(entry, 'MultiTaskDynamics', RecordingDynamics)
    monkeypatch.setattr(runtime, 'prepare_pretrain_output', lambda *a: tmp_path / 'dynamics')
    monkeypatch.setattr(entry.ptu.RunningMeanStd, 'update', lambda *a: pytest.fail('Unexpected refit'))
    entry.experiment(variant, seed=0)
    stats = load_normalizer(target / 'observation_normalizer.npz')
    expected = (np.array([0, 0], dtype=np.float32) - stats['mean']) / np.sqrt(stats['var'] + 1e-8)
    np.testing.assert_array_equal(observed[0]['observations'][0], expected)
    assert len(observed) == 10
    assert file_hash(tmp_path / 'dynamics/observation_normalizer.npz') == file_hash(target / 'observation_normalizer.npz')


def test_encoder_pretrain_and_shared_fixed_context_evaluation(published, tmp_path, monkeypatch):
    import torch
    import pretrain_encoder_decoder as entry
    import shared_protocol.runtime as runtime
    import evaluate_fixed_context as evaluation
    from rlkit.torch.networks import FlattenMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    cfg, target = published
    variant = variant_for(cfg, target)
    # Deliberately tiny TEST ONLY optimization; formal configs are untouched.
    variant.update(num_iters=2, net_size=8, seed=0)
    variant['util_params']['use_gpu'] = False
    variant['algo_params'].update(relabel_data_ratio=0., replay_buffer_size=250,
        embedding_batch_size=8, meta_batch=2)
    pretrained = tmp_path / 'pretrain'
    monkeypatch.setattr(runtime, 'prepare_pretrain_output', lambda *a: pretrained)
    monkeypatch.setattr(entry.ptu.RunningMeanStd, 'update', lambda *a: pytest.fail('Unexpected refit'))
    entry.experiment(variant, seed=0)
    assert read_json(pretrained / 'pretrain_metadata.json')['num_iters'] == 2
    assert file_hash(pretrained / 'observation_normalizer.npz') == file_hash(target / 'observation_normalizer.npz')
    run = tmp_path / 'run'
    run.mkdir()
    for name in CORE_FILES:
        shutil.copyfile(pretrained / name, run / name)
    for name in ('context_encoder', 'context_decoder'):
        shutil.copyfile(pretrained / (name + '.pth'), run / (name + '_itr_499.pth'))
    latent = variant['latent_size']
    models = dict(policy=TanhGaussianPolicy(hidden_sizes=[8]*3, obs_dim=2+latent,
                                             latent_dim=latent, action_dim=2),
                  qf1=FlattenMlp(hidden_sizes=[8]*3, input_size=4+latent, output_size=1),
                  qf2=FlattenMlp(hidden_sizes=[8]*3, input_size=4+latent, output_size=1))
    for name, model in models.items():
        torch.save(model.state_dict(), run / (name + '_itr_499.pth'))
    write_json(run / 'variant.json', variant)
    out = tmp_path / 'diagnostics'
    evaluation.main(['--log-dir', str(run), '--epochs', '499', '--offline-only',
        '--context-size', '4', '--probe-size', '4', '--context-repeats', '1', '--eval-seeds', '0',
        '--output-dir', str(out)])
    result = read_json(out / 'fixed_context_results.json')
    assert result['observation_normalizer']['shared_artifact_hashes'] == variant['shared_artifact_hashes']


def test_focal_consumer_compatibility(published, tmp_path):
    """Optional cross-repository test against the actual consumer, not a schema mock."""
    focal = Path(os.environ.get('FOCAL_SHARED_ROOT', ROOT.parent / 'FOCAL'))
    if not (focal / 'focal_shared/data.py').exists():
        pytest.skip('Set FOCAL_SHARED_ROOT to run the independent consumer check')
    cfg, target = published
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-B', str(focal / 'scripts/shared_protocol.py'), 'prepare',
        '--env', 'point-robot', '--reference-root', str(ROOT), '--data-dir', cfg['data_dir'],
        '--artifacts', str(tmp_path / 'focal_import'), '--data-actions', 'normalized',
        '--normalizer', str(target / 'observation_normalizer.npz'),
        '--task-manifest', str(target / 'task_manifest.json')]
    result = subprocess.run(command, cwd=focal, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    imported = tmp_path / 'focal_import'
    for f in ('task_manifest.json', 'observation_normalizer.npz'):
        assert file_hash(target / f) == file_hash(imported / f)
    assert digest(read_json(target / 'dataset_manifest.json')) == digest(read_json(imported / 'dataset_manifest.json'))
    code = '''import sys
from focal_shared.data import load_prepared
from focal_shared.environments import make_env, export_or_check_tasks
from focal_shared.common import read_json
cfg=read_json(sys.argv[1]); cfg['data_dir']=sys.argv[2]
env=make_env(cfg,sys.argv[3]); export_or_check_tasks(env,cfg,sys.argv[4]+'/task_manifest.json')
pools,stats=load_prepared(cfg,env,sys.argv[4]); assert stats['count']==2000.0001
assert len(pools)==20; env.close(); print('actual FOCAL load_prepared passed')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code,
        str(focal / 'configs/shared_protocol/point-robot.json'), cfg['data_dir'], str(ROOT), str(target)],
        cwd=focal, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
