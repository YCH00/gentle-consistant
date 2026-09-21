"""Exercise production GENTLE methods without importing the MuJoCo runtime.

Only the class definitions are compiled from their source; all method bodies,
the task encoder/policy/critics, and the semantic sampler execute normally.
The environment, offline data source, and logging are the test boundaries.
"""

import ast
import copy
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet

from configs.default import default_config
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.agent import Agent
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.task_interpolation import SemanticTaskInterpolator


ROOT = Path(__file__).resolve().parents[1]


def load_production_classes():
    path = ROOT / 'rlkit/torch/algo/gentle.py'
    parsed = ast.parse(path.read_text(encoding='utf-8'))
    classes = [node for node in parsed.body if isinstance(node, ast.ClassDef)]
    namespace = dict(
        torch=torch, np=np, F=F, ptu=ptu, OrderedDict=OrderedDict,
        optim=torch.optim, Dirichlet=Dirichlet,
        OfflineMetaRLAlgorithm=object, SemanticTaskInterpolator=SemanticTaskInterpolator,
        create_stats_ordered_dict=lambda name, values: {name: float(np.mean(values))},
    )
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['GENTLE'], namespace['VirtualTaskReplayBuffer']


GENTLE, VirtualTaskReplayBuffer = load_production_classes()


class ToyEncoder(torch.nn.Module):
    output_size = 2

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, context):
        reward = context[..., 2:3]
        return torch.cat([self.scale * reward, reward * 0], dim=-1)


class ToyDecoder(torch.nn.Module):
    def __init__(self, dynamics=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.dynamics = dynamics

    def forward(self, obs, actions, z):
        reward = self.scale * (z[..., :1] + z[..., 1:2] * obs)
        if not self.dynamics:
            return reward
        return torch.cat([reward, obs + z[..., :1] * actions], dim=-1)


def make_algorithm(dynamics=False, enabled=True):
    torch.manual_seed(20)
    np.random.seed(20)
    ptu.set_gpu_mode(False)
    algo = GENTLE.__new__(GENTLE)
    for key, value in default_config['algo_params'].items():
        setattr(algo, key, copy.deepcopy(value))
    algo.env = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(1,)),
        action_space=SimpleNamespace(shape=(1,)),
    )
    algo.obs_dim = algo.action_dim = 1
    algo.latent_dim = 2
    algo.train_tasks = np.array([0, 1])
    algo.use_next_obs_in_context = dynamics
    algo.batch_size = algo.embedding_batch_size = 16
    algo.n_vt = 3 if enabled else 0
    algo.virtual_task_generation_mode = 'semantic'
    algo.virtual_transition_batch_size = 16
    algo.virtual_transition_loss_weight = 0.5
    algo._virtual_transition_recon_weight = 1.0
    algo._virtual_transition_recon_loss_ema = None
    algo.virtual_transition_log_nearest_distance = True
    algo.virtual_transition_use_cycle_weight = True
    algo.consistency_use_policy_relabel_data = True
    algo.max_action = 1.0
    algo.itr = algo._num_steps = 0
    algo.loss = {}
    algo.eval_statistics = None
    encoder = ToyEncoder()
    policy = TanhGaussianPolicy(hidden_sizes=[8], obs_dim=3, latent_dim=2, action_dim=1)
    algo.agent = Agent(2, encoder, policy, **{
        **default_config['algo_params'], 'use_next_obs_in_context': dynamics,
    })
    algo.context_decoder = ToyDecoder(dynamics)
    algo.qf1 = FlattenMlp(hidden_sizes=[8], input_size=4, output_size=1)
    algo.qf2 = FlattenMlp(hidden_sizes=[8], input_size=4, output_size=1)
    algo.target_qf1, algo.target_qf2 = copy.deepcopy(algo.qf1), copy.deepcopy(algo.qf2)
    for name, module in [('context', encoder), ('policy', policy), ('qf1', algo.qf1), ('qf2', algo.qf2)]:
        setattr(algo, name + '_optimizer', torch.optim.Adam(module.parameters(), lr=1e-3))
    algo.virtual_transition_buffer = VirtualTaskReplayBuffer(512, 1, 1, 2)
    options = dict(
        virtual_semantic_probe_batch_size=32, virtual_semantic_support_radius=1.0,
        virtual_semantic_min_shared_probes=4, virtual_semantic_path_steps=3,
        virtual_semantic_dynamics_weight=1.0 if dynamics else 0.0,
    )
    algo.virtual_semantic_probe_batch_size = 32
    algo.virtual_semantic_refresh_interval = 100
    algo._semantic_refresh_step = None
    algo._semantic_interpolator = SemanticTaskInterpolator(algo.context_decoder, dynamics, **options)
    task_z = torch.tensor([[0.2, 0.0], [0.8, 0.0]])
    data_obs = torch.rand(2, 128, 1) * 2 - 1
    data_actions = torch.rand(2, 128, 1) * 2 - 1
    z = task_z[:, None, :].expand(-1, 128, -1)
    prediction = algo.context_decoder(data_obs, data_actions, z)
    data = [data_obs, data_actions, prediction[..., :1],
            prediction[..., 1:] if dynamics else data_obs + data_actions,
            torch.zeros(2, 128, 1)]
    calls = []

    def transitions(indices, b_size):
        calls.append((tuple(indices), b_size))
        rows = torch.randint(128, (len(indices), b_size))
        tasks = torch.as_tensor(indices)[:, None].expand_as(rows)
        return [field[tasks, rows].detach() for field in data]

    def context(indices, b_size=None):
        batch = transitions(indices, b_size or algo.embedding_batch_size)
        return torch.cat(batch[:4] if dynamics else batch[:3], dim=-1)

    algo.sample_transition_batch = transitions
    algo.sample_context = context
    algo.sample_sac = lambda indices: transitions(indices, algo.batch_size)
    return algo, calls


@pytest.mark.parametrize('dynamics', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_full_training_step_with_real_policy_critic_and_encoder(dynamics, enabled):
    algo, _ = make_algorithm(dynamics, enabled)
    before = copy.deepcopy(algo.context_decoder.state_dict())
    result = algo._take_step([0, 1], algo.sample_context([0, 1], 16))
    assert result[0].shape == (2, 2)
    assert np.isfinite(result[0]).all()
    for key in ('qf_loss', 'policy_loss', 'consistency_loss', 'encoder_total_loss'):
        assert np.isfinite(algo.loss[key]), key
    assert algo.loss['num_virtual_tasks'] == (3 if enabled else 0)
    assert algo.eval_statistics['num_virtual_tasks'] == (3 if enabled else 0)
    if enabled:
        assert algo.eval_statistics['virtual_semantic_edges'] > 0
    assert algo.virtual_transition_buffer.size() == (48 if enabled else 0)
    for key, parameter in algo.context_decoder.state_dict().items():
        torch.testing.assert_close(parameter, before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in algo.context_decoder.parameters())


@pytest.mark.parametrize('dynamics', [False, True])
def test_virtual_transition_and_cycle_use_exact_support_inputs(dynamics):
    algo, _ = make_algorithm(dynamics)
    z, support = algo._sample_virtual_task_embeddings(16, return_metadata=True)
    assert support is not None

    def forbidden(*args, **kwargs):
        raise AssertionError('virtual data must not resample unrelated real tasks')

    algo.sample_transition_batch = forbidden
    algo.sample_context = forbidden
    algo.virtual_transition_log_nearest_distance = False
    algo.add_virtual_transitions_to_buffer(z, 16, support_batch=support)
    context = algo._build_consistency_context(z, 16, support_batch=support)
    torch.testing.assert_close(context[..., :1], support['observations'])
    torch.testing.assert_close(context[..., 1:2], support['actions'])
    count = z.shape[0] * 16
    np.testing.assert_allclose(algo.virtual_transition_buffer._observations[:count],
                               support['observations'].reshape(count, -1).numpy())
    np.testing.assert_allclose(algo.virtual_transition_buffer._actions[:count],
                               support['actions'].reshape(count, -1).numpy())
    if not dynamics:
        np.testing.assert_allclose(algo.virtual_transition_buffer._next_obs[:count],
                                   support['next_observations'].reshape(count, -1).numpy())
    assert (algo.virtual_transition_buffer._weights[:count] <=
            support['quality_weights'].repeat_interleave(16, dim=0).numpy() + 1e-6).all()
    loss = algo._compute_consistency_loss(z, 16, support_batch=support)
    loss.backward()
    assert algo.agent.context_encoder.scale.grad is not None


def test_geometry_cache_refreshes_by_training_step_and_explicit_request():
    algo, calls = make_algorithm()
    algo._sample_virtual_task_embeddings(16)
    first_calls = len(calls)
    algo._num_steps = 99
    algo._sample_virtual_task_embeddings(8)
    assert len(calls) == first_calls
    algo._num_steps = 100
    algo._sample_virtual_task_embeddings(16)
    assert len(calls) > first_calls
    assert algo._semantic_interpolator.stats['virtual_semantic_refresh_count'] == 2
    algo._sample_virtual_task_embeddings(16, force_refresh=True)
    assert algo._semantic_interpolator.stats['virtual_semantic_refresh_count'] == 3


def load_export_sampler(algo):
    path = ROOT / 'export_virtual_embeddings.py'
    parsed = ast.parse(path.read_text(encoding='utf-8'))
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'sample_virtual_embeddings')
    namespace = dict(
        np=np, torch=torch, ptu=ptu, Dirichlet=Dirichlet,
        SemanticTaskInterpolator=SemanticTaskInterpolator,
        sample_context_batch=lambda buffer, tasks, size: algo.sample_transition_batch(tasks, size),
        build_context=lambda obs, actions, rewards, next_obs, use_next:
            torch.cat([obs, actions, rewards] + ([next_obs] if use_next else []), dim=-1),
        infer_task_embedding=lambda context_encoder, context, **kwargs:
            context_encoder(context).mean(dim=1),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['sample_virtual_embeddings']


@pytest.mark.parametrize('mode', ['semantic', 'local', 'global', 'gaussian'])
def test_export_dispatch_uses_saved_generation_mode(mode):
    algo, _ = make_algorithm()
    export = load_export_sampler(algo)
    result = export(
        None, [0, 1], algo.agent.context_encoder, 2, False, False,
        n_vt=3, mixing_tasks=2, beta=1.0, batch_size=16, n_points=2,
        generation_mode=mode, context_decoder=algo.context_decoder,
        algo_params={'virtual_semantic_probe_batch_size': 32, 'virtual_semantic_path_steps': 0},
    )
    assert result.shape == (1, 6, 2)
    assert np.isfinite(result).all()


def test_semantic_export_does_not_fall_back_if_decoder_missing():
    algo, _ = make_algorithm()
    with pytest.raises(ValueError, match='decoder'):
        load_export_sampler(algo)(
            None, [0, 1], algo.agent.context_encoder, 2, False, False,
            n_vt=3, mixing_tasks=2, beta=1, batch_size=16, n_points=2,
            generation_mode='semantic',
        )


@pytest.mark.parametrize('mode', ['local', 'global', 'gaussian'])
def test_legacy_samplers_still_support_training_update(mode):
    algo, _ = make_algorithm()
    algo.virtual_task_generation_mode = mode
    algo._semantic_interpolator = None
    algo.M, algo.beta = 2, 1.0
    algo._take_step([0, 1], algo.sample_context([0, 1], 16))
    assert algo.loss['num_virtual_tasks'] == 3
    assert np.isfinite(algo.loss['policy_total_loss'])


def test_training_continues_and_logs_when_no_semantic_edge_is_supported():
    algo, _ = make_algorithm()
    algo._semantic_interpolator = SemanticTaskInterpolator(
        algo.context_decoder, False, virtual_semantic_support_radius=1e-8,
    )
    algo._take_step([0, 1], algo.sample_context([0, 1], 16))
    assert algo.loss['num_virtual_tasks'] == 0
    assert algo.virtual_transition_buffer.size() == 0
    assert algo.eval_statistics['virtual_semantic_edges'] == 0
    assert algo.eval_statistics['virtual_semantic_rejected_no_path'] > 0
    assert np.isfinite(algo.loss['policy_total_loss'])


@pytest.mark.parametrize('use_quality', [False, True])
def test_semantic_quality_switch_controls_replay_weights_only(use_quality):
    algo, _ = make_algorithm()
    algo.virtual_transition_use_cycle_weight = False
    algo.virtual_transition_use_semantic_weight = use_quality
    z, support = algo._sample_virtual_task_embeddings(16, return_metadata=True)
    # Non-unit values distinguish the switch from perfect endpoint predictions.
    support['quality_weights'] = torch.tensor([[0.4], [0.6], [0.8]])
    algo.add_virtual_transitions_to_buffer(z, 16, support_batch=support)
    expected = support['quality_weights'] if use_quality else torch.ones(3, 1)
    np.testing.assert_allclose(algo.virtual_transition_buffer._weights[:48],
                               expected.repeat_interleave(16, dim=0).numpy())
    torch.testing.assert_close(support['quality_weights'], torch.tensor([[0.4], [0.6], [0.8]]))
    np.testing.assert_allclose(algo.virtual_transition_buffer._observations[:48],
                               support['observations'].reshape(48, 1).numpy())


@pytest.mark.parametrize('mode,steps', [('global', 0), ('semantic', 0), ('semantic', 8)])
@pytest.mark.parametrize('dynamics', [False, True])
def test_ablation_training_uses_only_linear_sample_weight(mode, steps, dynamics):
    algo, _ = make_algorithm(dynamics)
    algo.virtual_task_generation_mode = mode
    algo.virtual_transition_use_cycle_weight = False
    algo.virtual_transition_use_semantic_weight = False
    algo.virtual_transition_use_recon_weight = False
    algo.virtual_transition_train_policy_bc = False
    algo.virtual_transition_train_policy_q = True
    algo.virtual_transition_weight_schedule = 'linear_decay'
    algo.virtual_transition_weight_decay_start_itr = 2
    algo.virtual_transition_weight_decay_end_itr = 4
    algo.virtual_transition_final_loss_weight = 0.0
    if mode == 'global':
        algo._semantic_interpolator = None
        algo.M, algo.beta = 2, 1.0
    else:
        algo._semantic_interpolator.options['virtual_semantic_path_steps'] = steps
    for iteration, expected_weight in [(0, 0.5), (2, 0.5), (3, 0.25), (4, 0.0)]:
        algo.itr = iteration
        algo.eval_statistics = None
        algo._take_step([0, 1], algo.sample_context([0, 1], 16))
        assert algo.loss['virtual_transition_effective_weight_sample_mean'] == pytest.approx(expected_weight)
        assert algo.eval_statistics['virtual_transition_recon_weight_current'] == 1.0
        assert algo.eval_statistics['virtual_transition_use_semantic_weight'] == 0
        assert algo.eval_statistics['virtual_transition_use_recon_weight'] == 0
        assert algo.eval_statistics['virtual_transition_use_cycle_weight'] == 0
        assert algo.loss['virtual_transition_policy_bc_batch_size'] == 0
        assert algo.loss['virtual_consistency_fraction'] == pytest.approx(3 / 5)
        combined = (2 * algo.loss['real_consistency_loss'] + 3 * algo.loss['virtual_consistency_loss']) / 5
        assert algo.loss['consistency_loss'] == pytest.approx(combined, abs=1e-7)
        assert algo.eval_statistics['virtual_diagnostic_training_calls'] == 1
        if expected_weight == 0:
            assert algo.loss['num_virtual_transitions_added'] == 0
            assert algo.loss['num_virtual_tasks'] == 3  # consistency still uses virtual tasks
        algo._num_steps += 1


def test_diagnostic_iteration_means_include_skips_and_reset():
    algo, _ = make_algorithm()
    algo._take_step([0, 1], algo.sample_context([0, 1], 16))
    algo._sample_virtual_task_embeddings = lambda *a, **kw: (None, None)
    algo._take_step([0, 1], algo.sample_context([0, 1], 16))
    assert algo.eval_statistics['num_virtual_tasks_itr_mean'] == 1.5
    assert algo.eval_statistics['virtual_consistency_fraction_itr_mean'] == pytest.approx(0.3)
    assert algo.eval_statistics['virtual_task_generation_skipped_itr_mean'] == 0.5
    assert algo.eval_statistics['num_virtual_transitions_added_itr_total'] == 48
    algo.itr += 1
    algo.eval_statistics = None
    algo._take_step([0, 1], algo.sample_context([0, 1], 16))
    assert algo.eval_statistics['num_virtual_tasks_itr_mean'] == 0
    assert algo.eval_statistics['virtual_task_generation_skipped_itr_mean'] == 1
    assert algo.eval_statistics['num_virtual_transitions_added_itr_total'] == 0
    assert algo.eval_statistics['virtual_diagnostic_training_calls'] == 1


def test_per_task_consistency_logging_preserves_loss_and_gradient():
    algo, _ = make_algorithm()
    z, support = algo._sample_virtual_task_embeddings(16, return_metadata=True)
    scalar = algo._compute_consistency_loss(z, 16, support_batch=support)
    per_task = algo._compute_consistency_loss(z, 16, support_batch=support, return_task_losses=True)
    torch.testing.assert_close(scalar, per_task.mean())
    parameter = algo.agent.context_encoder.scale
    grad_scalar, = torch.autograd.grad(scalar, parameter)
    grad_tasks, = torch.autograd.grad(per_task.mean(), parameter)
    torch.testing.assert_close(grad_scalar, grad_tasks)
