"""Execute the complete checkpoint CLI with real networks and a small environment."""
import ast
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

import evaluate_fixed_context as cli
from configs.default import default_config
from rlkit.data_management.fixed_context_data import compute_observation_stats, load_offline_pools
from rlkit.torch.autoencoder import MlpDecoder, MlpEncoder
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy
from test_fixed_context_data import save_trajectory


ROOT = Path(__file__).resolve().parents[1]


class ToyEnvironment:
    instances = []

    def __init__(self, **kwargs):
        self.observation_space = SimpleNamespace(shape=(2,))
        self.action_space = SimpleNamespace(shape=(1,))
        self.task = 0
        self.closed = False
        self.seed(0)
        self.instances.append(self)

    def seed(self, value):
        self.rng = np.random.RandomState(value)

    def get_all_task_idx(self):
        return [0, 1]

    def reset_task(self, task):
        self.task = task
        self.reset()

    def reset(self):
        self.position = self.rng.uniform(-1, 1)
        self.steps = 0
        return np.array([self.position, self.position + 0.25])

    def step(self, action):
        self.position += float(action[0]) * 0.1
        self.steps += 1
        return np.array([self.position, self.position + 0.25]), -abs(self.position-self.task), self.steps == 3, {}

    def close(self):
        self.closed = True


class NormalizedToy:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, key):
        return getattr(self.env, key)

    def update_obs_mean_var(self, mean, var):
        self.mean, self.var = mean, var

    def reset(self):
        return (self.env.reset() - self.mean) / np.sqrt(self.var + 1e-8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return (obs - self.mean) / np.sqrt(self.var + 1e-8), reward, done, info


def fixture_run(tmp_path, monkeypatch, dynamics=False):
    env_module = ModuleType('rlkit.envs')
    env_module.ENVS = {'toy': ToyEnvironment}
    wrapper_module = ModuleType('rlkit.envs.wrappers')
    wrapper_module.NormalizedBoxEnv = NormalizedToy
    monkeypatch.setitem(sys.modules, 'rlkit.envs', env_module)
    monkeypatch.setitem(sys.modules, 'rlkit.envs.wrappers', wrapper_module)
    log = tmp_path / 'run'
    log.mkdir()
    data = tmp_path / 'data'
    for task in range(2):
        save_trajectory(data, task=task, count=20, base=task*30, epoch=10)
    variant = copy.deepcopy(default_config)
    variant.update(env_name='toy', env_params={'n_tasks': 2}, n_train_tasks=1, n_eval_tasks=1,
                   latent_size=2, net_size=8, seed=0)
    variant['algo_params'].update(data_dir=str(data), n_trj=1, train_epoch=10, eval_epoch=10,
        max_path_length=3, replay_buffer_size=20, use_next_obs_in_context=dynamics)
    (log / 'variant.json').write_text(json.dumps(variant))
    torch.manual_seed(10)
    models = dict(
        context_encoder=MlpEncoder(hidden_sizes=[8]*3, input_size=6 if dynamics else 4,
                                   output_size=2, output_activation=torch.tanh, batch_attention=False),
        context_decoder=MlpDecoder(hidden_size=8, num_hidden_layers=3, z_dim=2, action_dim=1,
                                   obs_dim=2, reward_dim=1, use_next_obs_in_context=dynamics),
        policy=TanhGaussianPolicy(hidden_sizes=[8]*3, obs_dim=4, latent_dim=2, action_dim=1),
        qf1=FlattenMlp(hidden_sizes=[8]*3, input_size=5, output_size=1),
        qf2=FlattenMlp(hidden_sizes=[8]*3, input_size=5, output_size=1))
    for epoch in (0, 2):
        with torch.no_grad():
            models['context_encoder'].last_fc.bias.fill_(epoch*0.2)
        for name, model in models.items():
            torch.save(model.state_dict(), log / ('%s_itr_%d.pth' % (name, epoch)))
    output = tmp_path / 'diagnostics'
    args = ['--log-dir', str(log), '--epochs', '0', '2', '--collector-epochs', '0', '2',
            '--output-dir', str(output), '--context-size', '2', '--context-repeats', '2',
            '--probe-size', '3', '--eval-seeds', '7', '8', '--save-trajectories']
    return log, data, output, args, variant


@pytest.mark.parametrize('dynamics', [False, True])
def test_complete_cli_real_checkpoints_fixed_sources_cache_and_diagnostics(tmp_path, monkeypatch, dynamics):
    log, data, output, args, variant = fixture_run(tmp_path, monkeypatch, dynamics)
    before = {p.name: cli.sha256(p) for p in log.iterdir()}
    cli.main(args)
    result_path = output / 'fixed_context_results.json'
    first = json.loads(result_path.read_text(encoding='utf-8'))
    assert len(first['records']) == 12  # 2 checkpoints, 3 sources, 2 independent contexts
    assert len(first['latent_statistics']) == 6
    assert first['observation_normalizer']['source'] == 'reconstructed_from_current_training_data'
    for r in first['records']:
        assert r['evaluation']['context_size'] == 2
        assert len(r['evaluation']['returns']) == 2
        assert r['fixed_probe']['count'] == 3
        assert 'q1_finite_horizon_mc_error' in r['evaluation']
        assert 'q1_finite_horizon_mc_error' not in r['fixed_probe']
        assert 'decoder_reward_error' in r['fixed_probe']
        assert ('decoder_next_observation_error' in r['fixed_probe']) == dynamics
        assert r['z_l2_from_first_checkpoint'] == 0 if r['epoch'] == 0 else r['z_l2_from_first_checkpoint'] > 0
        reference = next(x for x in first['records'] if x['epoch'] == 0 and x['source'] == 'offline'
                         and x['repeat'] == r['repeat'])
        assert r['reset_seeds'] == reference['reset_seeds']
    assert ToyEnvironment.instances[-1].closed
    assert {p.name: cli.sha256(p) for p in log.iterdir()} == before
    with np.load(output / 'trajectory_diagnostics.npz', allow_pickle=False) as stored:
        assert stored.files
        assert all(stored[key].dtype != object for key in stored.files)
    # Repeated evaluation uses the saved exact context bytes, not new collections.
    import rlkit.torch.fixed_context_eval as core
    def forbidden(*args, **kwargs):
        raise AssertionError('must reuse the saved context bank')
    monkeypatch.setattr(core, 'collect_zero_context', forbidden)
    cli.main(args)
    second = json.loads(result_path.read_text(encoding='utf-8'))
    assert first == second
    bank = output / 'context_bank'
    manifest = json.loads((bank / 'manifest.json').read_text())
    with np.load(bank / 'contexts.npz', allow_pickle=False) as stored:
        ctx = set(stored['offline_task_1_repeat_0__row_ids']) | set(stored['offline_task_1_repeat_1__row_ids'])
        assert not ctx.intersection(stored['probe_task_1__row_ids'])
    assert manifest['metadata']['max_path_length'] == 3


def test_cache_rejects_changed_context_protocol_or_checkpoint(tmp_path, monkeypatch):
    log, data, output, args, variant = fixture_run(tmp_path, monkeypatch)
    cli.main(args)
    with pytest.raises(ValueError, match='metadata differs'):
        cli.main(args + ['--context-size', '3'])
    variant['algo_params']['max_path_length'] = 4
    (log / 'variant.json').write_text(json.dumps(variant))
    with pytest.raises(ValueError, match='metadata differs'):
        cli.main(args)


def test_distinct_runs_can_share_the_exact_reference_collector_bank(tmp_path, monkeypatch):
    log, data, output, args, variant = fixture_run(tmp_path, monkeypatch)
    cli.main(args)
    other = tmp_path / 'other_run'
    shutil.copytree(log, other)
    other_output = tmp_path / 'other_diagnostics'
    import rlkit.torch.fixed_context_eval as core
    def forbidden(*args, **kwargs):
        raise AssertionError('cross-run comparison must reuse reference contexts')
    monkeypatch.setattr(core, 'collect_zero_context', forbidden)
    cli.main(args + ['--log-dir', str(other), '--collector-log-dir', str(log),
                    '--bank-dir', str(output / 'context_bank'), '--output-dir', str(other_output)])
    original = json.loads((output / 'fixed_context_results.json').read_text())
    shared = json.loads((other_output / 'fixed_context_results.json').read_text())
    assert original['records'] == shared['records']
    assert original['bank_archive_sha256'] == shared['bank_archive_sha256']


def test_modified_cache_is_rejected_before_evaluation(tmp_path, monkeypatch):
    log, data, output, args, variant = fixture_run(tmp_path, monkeypatch)
    cli.main(args)
    archive = output / 'context_bank/contexts.npz'
    archive.write_bytes(archive.read_bytes() + b'modified')
    with pytest.raises(ValueError, match='checksum'):
        cli.main(args)


def test_incomplete_bank_manifest_is_rejected(tmp_path, monkeypatch):
    log, data, output, args, variant = fixture_run(tmp_path, monkeypatch)
    cli.main(args)
    path = output / 'context_bank/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['entries'].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='complete source/task/repeat'):
        cli.main(args)


def test_missing_checkpoint_fails_without_substitution(tmp_path):
    with pytest.raises(FileNotFoundError, match='no fallback'):
        cli.checkpoint_paths(tmp_path, 499)


def test_help_does_not_import_numpy_torch_or_environment():
    source = "import sys, evaluate_fixed_context as m; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; assert 'rlkit.envs' not in sys.modules; m.main(['--help'])"
    result = subprocess.run([sys.executable, '-c', source], cwd=str(ROOT), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--bank-dir' in result.stdout


def test_collector_horizon_must_match(tmp_path, monkeypatch):
    *_, variant = fixture_run(tmp_path, monkeypatch)
    collector = copy.deepcopy(variant)
    collector['algo_params']['max_path_length'] += 1
    with pytest.raises(ValueError, match='max_path_length'):
        cli.validate_collector_variant(variant, collector)


def save_training_stats(log, stats):
    source = ROOT / 'train_gentle.py'
    function = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_save_observation_normalizer')
    namespace = dict(np=np, Path=Path)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    algo = SimpleNamespace(obs_normalizer=SimpleNamespace(**stats), train_tasks=[0], eval_tasks=[1])
    namespace['_save_observation_normalizer'](algo, log)


def test_saved_normalizer_used_exactly_and_changed_data_rejected(tmp_path, monkeypatch):
    log, data, output, args, variant = fixture_run(tmp_path, monkeypatch)
    pools, _ = load_offline_pools(data, [0], 1, 10, 2, 1)
    stats = compute_observation_stats(pools)
    save_training_stats(log, stats)
    resolved, source = cli.resolve_observation_stats(log, pools, [0], [1])
    for key in stats:
        np.testing.assert_array_equal(resolved[key], stats[key])
    assert source['source'] == 'saved_training_normalizer'
    cli.main(args + ['--offline-only'])
    result = json.loads((output / 'fixed_context_results.json').read_text())
    assert result['observation_normalizer']['sha256'] == cli.sha256(log / 'observation_normalizer.npz')
    assert len(result['records']) == 4
    pools[0]['observations'] += 50
    with pytest.raises(ValueError, match='no longer matches'):
        cli.resolve_observation_stats(log, pools, [0], [1])


def test_output_does_not_overwrite_training_run(tmp_path, monkeypatch):
    log, data, output, args, variant = fixture_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='outside'):
        cli.main(args + ['--output-dir', str(log / 'diagnostics')])


def test_sparse_rewards_fail_explicitly(tmp_path, monkeypatch):
    log, data, output, args, variant = fixture_run(tmp_path, monkeypatch)
    variant['algo_params']['sparse_rewards'] = True
    (log / 'variant.json').write_text(json.dumps(variant))
    with pytest.raises(ValueError, match='dense rewards'):
        cli.main(args)


def test_nonfinite_results_cannot_be_exported(tmp_path):
    with pytest.raises(ValueError):
        cli.write_json(tmp_path / 'out.json', dict(value=float('nan')))
