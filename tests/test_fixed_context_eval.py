import copy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from rlkit.torch.fixed_context_eval import (
    collect_zero_context, diagnose_fixed_batch, evaluate_fixed_context,
    infer_fixed_embedding, isolated_evaluation, reset_for_task, step_environment,
)
from rlkit.torch.sac.agent import _product_of_gaussians


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(0.8)
        self.calls = 0
        self.seen = []

    def forward(self, context):
        self.calls += 1
        self.seen.append(context.detach().clone())
        return self.dropout(context[..., :1]) * self.scale


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.inputs = []
        self.deterministic = []

    def forward(self, meta_size, batch_size, value, deterministic=False):
        assert meta_size == 1 and batch_size == len(value)
        self.inputs.append(value.detach().clone())
        self.deterministic.append(deterministic)
        action = value[:, -1:] * self.scale
        if not deterministic:
            action = action + torch.randn_like(action)
        return action, None


class Decoder(nn.Module):
    def __init__(self, dynamics=False):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(1.0))
        self.dynamics = dynamics

    def forward(self, obs, action, latent):
        reward = obs[:, :1] + self.offset
        return torch.cat((reward, obs + 1.0), dim=1) if self.dynamics else reward


class Critic(nn.Module):
    def __init__(self, value=2.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, meta_size, batch_size, obs, action, latent):
        assert meta_size == 1 and batch_size == len(obs)
        return torch.ones_like(obs[:, :1]) * self.value


class Environment:
    def __init__(self, terminal_at=3, timeout_at=None, modern=False, random_initial=False):
        self.terminal_at = terminal_at
        self.timeout_at = timeout_at
        self.modern = modern
        self.random_initial = random_initial
        self.seed_calls = []
        self.reset_calls = 0
        self.task_calls = []
        self.actions = []

    def seed(self, seed):
        self.seed_calls.append(seed)
        self.rng = np.random.RandomState(seed)

    def reset_task(self, task_idx):
        self.task_calls.append(task_idx)

    def reset(self, seed=None):
        self.reset_calls += 1
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.position = float(self.rng.uniform()) if self.random_initial else 0.0
        self.steps = 0
        obs = np.array([self.position], dtype=np.float32)
        return (obs, {}) if self.modern else obs

    def step(self, action):
        self.actions.append(action.copy())
        random.random()
        np.random.random()
        torch.rand(1)
        self.steps += 1
        self.position += 1.0
        obs = np.array([self.position], dtype=np.float32)
        terminal = self.terminal_at is not None and self.steps == self.terminal_at
        timeout = self.timeout_at is not None and self.steps == self.timeout_at
        if self.modern:
            return obs, self.position, terminal, timeout, {}
        return obs, self.position, terminal or timeout, {'TimeLimit.truncated': timeout}


def run(env=None, **kwargs):
    options = dict(env=env or Environment(), policy=Policy(), encoder=Encoder(),
                   context=torch.tensor([[[2.0], [4.0]]]), task_idx=7,
                   reset_seeds=[10, 20], max_path_length=10, discount=0.5,
                   reward_scale=2.0, latent_dim=1)
    options.update(kwargs)
    return evaluate_fixed_context(**options)


def test_context_encoded_once_and_frozen_for_all_rollouts():
    encoder, policy, env = Encoder(), Policy(), Environment()
    context = torch.tensor([[[2.0], [4.0]]])
    before = context.clone()
    summary, paths = run(env, encoder=encoder, policy=policy, context=context)
    assert encoder.calls == 1
    torch.testing.assert_close(encoder.seen[0], before)
    torch.testing.assert_close(context, before)
    assert summary['context_size'] == 2 and summary['z'] == [[3.0]]
    assert all(policy.deterministic)
    assert all(item[0, -1].item() == 3.0 for item in policy.inputs)
    assert env.seed_calls == [10, 20] and env.task_calls == [7, 7]
    assert env.reset_calls == 2
    assert [len(item['rewards']) for item in paths] == [3, 3]
    json.dumps(summary, allow_nan=False)


def test_legacy_done_at_internal_limit_is_not_guaranteed_true_termination():
    class AmbiguousEnvironment(Environment):
        def step(self, action):
            obs, reward, done, _ = super().step(action)
            return obs, reward, done, {}
    summary, _ = run(AmbiguousEnvironment(terminal_at=3), max_path_length=3)
    for rollout in summary['per_rollout']:
        assert rollout['terminated']  # preserve the environment's original done flag
        assert rollout['horizon_reached']
        assert rollout['termination_cause_ambiguous']
    assert 'Legacy done' in summary['termination_note']


def test_ib_pooling_exactly_matches_agent_with_variance_clamping():
    class IBEncoder(nn.Module):
        def forward(self, value):
            return value
    mus = torch.tensor([[2., -1.], [6., 4.]])
    logits = torch.tensor([[0.2, -100.], [1.7, 1.2]])
    context = torch.cat((mus, logits), dim=1)[None]
    actual = infer_fixed_embedding(IBEncoder(), context, 2, True)
    expected, _ = _product_of_gaussians(mus, torch.nn.functional.softplus(logits))
    torch.testing.assert_close(actual, expected[None], rtol=0, atol=0)
    assert not actual.requires_grad


def test_returns_mc_scaling_and_critic_errors_are_hand_computable():
    summary, paths = run(qf1=Critic(2.0), qf2=Critic(1.0), decoder=Decoder())
    assert summary['returns'] == [6.0, 6.0]
    assert summary['per_rollout'][0]['discounted_return_raw'] == 2.75
    assert summary['per_rollout'][0]['discounted_return_scaled'] == 5.5
    np.testing.assert_allclose(paths[0]['discounted_raw_returns'][:, 0], [2.75, 3.5, 3.0])
    np.testing.assert_allclose(paths[0]['discounted_scaled_returns'][:, 0], [5.5, 7.0, 6.0])
    errors = summary['q1_finite_horizon_mc_error']
    assert errors['bias'] == pytest.approx(-12.5 / 3)
    assert errors['mae'] == pytest.approx(12.5 / 3)
    assert errors['rmse'] == pytest.approx(np.sqrt((3.5**2 + 5**2 + 4**2) / 3))
    assert summary['decoder_reward_error'] == dict(bias=0.0, mae=0.0, rmse=0.0)
    np.testing.assert_array_equal(paths[0]['q_min'], 1.0)


@pytest.mark.parametrize('modern', [False, True])
@pytest.mark.parametrize('end_type', ['terminal', 'timeout', 'horizon'])
def test_termination_and_timeouts_are_distinct(modern, end_type):
    env = Environment(terminal_at=2 if end_type == 'terminal' else None,
                      timeout_at=2 if end_type == 'timeout' else None, modern=modern)
    summary, paths = run(env, max_path_length=2)
    assert summary['per_rollout'][0]['terminated'] == (end_type == 'terminal')
    assert summary['per_rollout'][0]['truncated'] == (end_type != 'terminal')
    assert bool(paths[0]['terminals'][-1, 0]) == (end_type == 'terminal')
    assert bool(paths[0]['timeouts'][-1, 0]) == (end_type != 'terminal')


def test_new_gym_reset_without_legacy_seed():
    env = Environment(modern=True, random_initial=True)
    env.seed = None
    _, first = run(env, reset_seeds=[51])
    _, second = run(env, reset_seeds=[51])
    np.testing.assert_array_equal(first[0]['observations'], second[0]['observations'])


def test_same_reset_seeds_make_comparisons_fair():
    _, first = run(Environment(random_initial=True), reset_seeds=[51, 52])
    _, second = run(Environment(random_initial=True), reset_seeds=[51, 52])
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left['observations'], right['observations'])
    assert first[0]['observations'][0, 0] != first[1]['observations'][0, 0]


@pytest.mark.parametrize('raises', [False, True])
def test_rng_modes_parameters_and_gradients_restored_even_after_exception(raises):
    encoder, policy = Encoder(), Policy()
    encoder.train()
    encoder.dropout.eval()
    policy.eval()
    modules = [encoder, policy]
    parameters = [p for module in modules for p in module.parameters()]
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    snapshots = [copy.deepcopy(module.state_dict()) for module in modules]
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    if raises:
        class BrokenEnv(Environment):
            def step(self, action):
                super().step(action)
                raise RuntimeError('environment failed')
        with pytest.raises(RuntimeError, match='environment failed'):
            run(BrokenEnv(), encoder=encoder, policy=policy)
    else:
        run(encoder=encoder, policy=policy)
    assert random.getstate() == py_state
    assert np.random.get_state()[0] == np_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert encoder.training and not encoder.dropout.training and not policy.training
    for module, snapshot in zip(modules, snapshots):
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)
    for parameter in parameters:
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))


@pytest.mark.parametrize('dynamics', [False, True])
def test_fixed_batch_decoder_targets_absolute_next_state_and_no_offline_mc(dynamics):
    obs = torch.tensor([[0.0], [1.0], [2.0]])
    summary, arrays = diagnose_fixed_batch(Decoder(dynamics), Critic(), Critic(3.0),
        obs, torch.zeros_like(obs), obs + 1.0, obs + 1.0, torch.ones(1, 1), dynamics)
    assert summary['decoder_reward_error']['rmse'] == 0
    if dynamics:
        assert summary['decoder_next_observation_error']['rmse'] == 0
        np.testing.assert_array_equal(arrays['decoder_predictions'][:, 1:], obs.numpy() + 1.0)
    assert not any('mc' in key.lower() for key in summary)
    assert summary['q_disagreement'] == 1.0


def test_evaluation_dynamics_decoder_metrics():
    summary, _ = run(decoder=Decoder(True), use_next_obs_in_context=True)
    assert summary['decoder_next_observation_error']['mae'] == 0


def test_zero_context_collects_exact_fresh_rows_and_reproducible_stochastic_actions():
    policy, env = Policy(), Environment(terminal_at=2)
    first = collect_zero_context(env, policy, 7, 51, 5, 10, 1)
    second = collect_zero_context(Environment(terminal_at=2), Policy(), 7, 51, 5, 10, 1)
    assert len(first['observations']) == 5
    assert env.seed_calls == [51, 52, 53]
    assert policy.deterministic == [False] * 5
    assert all(item[0, -1].item() == 0 for item in policy.inputs)
    assert len(np.unique(first['actions'])) == 5
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(first['terminals'][:, 0], [False, True, False, True, False])
    np.testing.assert_array_equal(first['timeouts'][:, 0], [False, False, False, False, True])


def test_isolation_public_list_api_and_seed_restores_global_state():
    module = Encoder()
    before = torch.get_rng_state()
    with isolated_evaluation([module], seed=43):
        assert not module.training
        first = torch.rand(3)
    with isolated_evaluation([module], seed=43):
        second = torch.rand(3)
    torch.testing.assert_close(first, second)
    assert torch.equal(before, torch.get_rng_state())
    assert module.training


@pytest.mark.parametrize('context', [torch.zeros(2, 3, 1), torch.empty(1, 0, 1),
                                    torch.zeros(3, 1), torch.full((1, 3, 1), float('nan'))])
def test_reject_invalid_context(context):
    with pytest.raises(ValueError):
        infer_fixed_embedding(Encoder(), context, 1)


@pytest.mark.parametrize('options', [dict(reset_seeds=[]), dict(reset_seeds=[-1]),
    dict(reset_seeds=[1.5]), dict(max_path_length=0), dict(discount=1.1),
    dict(discount=float('nan')), dict(reward_scale=0), dict(latent_dim=2)])
def test_reject_invalid_evaluation_options(options):
    with pytest.raises(ValueError):
        run(**options)


def test_reject_nonfinite_decoder_and_bad_transition_shapes():
    obs = torch.ones(3, 1)
    with pytest.raises(ValueError, match='transition shapes'):
        diagnose_fixed_batch(None, None, None, obs, obs[:2], obs, obs, torch.ones(1, 1))
    decoder = Decoder()
    decoder.offset.data.fill_(float('inf'))
    with pytest.raises(ValueError, match='nonfinite'):
        diagnose_fixed_batch(decoder, None, None, obs, obs, obs, obs, torch.ones(1, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_rng_restored():
    torch.cuda.init()
    states = torch.cuda.get_rng_state_all()
    with isolated_evaluation([], seed=29):
        torch.rand(10, device='cuda')
    assert all(torch.equal(left, right) for left, right in zip(states, torch.cuda.get_rng_state_all()))
