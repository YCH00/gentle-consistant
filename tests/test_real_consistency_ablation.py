"""Behavioral checks for real-only consistency ablations and old virtual paths."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from click.testing import CliRunner

from test_interpolation_ablation import training_entrypoint
from test_semantic_integration import GENTLE, make_algorithm


ROOT = Path(__file__).resolve().parents[1]


def forbidden(*args, **kwargs):
    raise AssertionError('paired replay must not obtain unrelated or policy actions')


class InputSensitiveDecoder(torch.nn.Module):
    """A trainable decoder deliberately detects accidental gradient leakage."""

    def __init__(self, dynamics):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.7))
        self.dynamics = dynamics

    def forward(self, observations, actions, z):
        reward = self.scale * (observations + 2 * actions + z[..., :1])
        if self.dynamics:
            return torch.cat([reward, observations + actions + z[..., 1:2]], -1)
        return reward


@pytest.mark.parametrize('dynamics', [False, True])
def test_paired_replay_preserves_rows_and_uses_decoder_labels_without_generator_gradients(dynamics):
    algo, _ = make_algorithm(dynamics, enabled=False)
    algo.context_decoder = InputSensitiveDecoder(dynamics)
    algo.agent.context_encoder.scale.data.fill_(0.4)
    indices = [1, 0, 1]
    observations = torch.arange(12., dtype=torch.float32).reshape(3, 4, 1) / 10
    actions = 0.25 - observations / 3
    fields = [observations, actions, torch.full_like(observations, -900.)]
    if dynamics:
        fields.append(torch.full_like(observations, 800.))
    replay_context = torch.cat(fields, -1).requires_grad_()
    calls = []

    def sample_context(task_indices, b_size=None):
        calls.append((list(task_indices), b_size))
        return replay_context

    algo.sample_context = sample_context
    algo._sample_mismatched_task_indices = forbidden
    algo.agent.policy.forward = forbidden
    task_z = torch.tensor([[0.2, 0.1], [0.8, 0.2], [0.3, 0.4]], requires_grad=True)
    generated = algo._build_consistency_context(
        task_z, 4, anchor_task_indices=indices, input_mode='paired_replay')
    assert calls == [(indices, 4)]
    assert generated.shape == replay_context.shape
    assert not generated.requires_grad
    torch.testing.assert_close(generated[..., :2], replay_context[..., :2])
    with torch.no_grad():
        expected_labels = algo.context_decoder(observations, actions, task_z[:, None, :].expand(-1, 4, -1))
    torch.testing.assert_close(generated[..., 2:], expected_labels)
    assert not torch.equal(generated[..., 2:], replay_context[..., 2:])

    encoder_before = algo.agent.context_encoder.scale.detach().clone()
    decoder_before = copy.deepcopy(algo.context_decoder.state_dict())
    loss = algo._compute_consistency_loss(
        task_z, 4, anchor_task_indices=indices, input_mode='paired_replay')
    loss.backward()
    assert algo.agent.context_encoder.scale.grad.abs().item() > 0
    assert task_z.grad is None and replay_context.grad is None
    assert all(parameter.grad is None for parameter in algo.context_decoder.parameters())
    assert all(parameter.grad is None for parameter in algo.agent.policy.parameters())
    algo.context_optimizer.step()
    assert not torch.equal(algo.agent.context_encoder.scale, encoder_before)
    for name, value in algo.context_decoder.state_dict().items():
        torch.testing.assert_close(value, decoder_before[name], rtol=0, atol=0)


@torch.no_grad()
def legacy_context_reference(algo, task_z, batch_size, anchor_task_indices=None,
                             use_policy_relabel_data=None):
    """Pre-ablation sampling order, including policy and cross-task RNG calls."""
    task_z = task_z.detach()
    count = task_z.size(0)
    if anchor_task_indices is None:
        anchor_task_indices = np.random.choice(algo.train_tasks, size=count, replace=True)
    else:
        anchor_task_indices = np.asarray(anchor_task_indices)
    anchor_context = algo.sample_context(anchor_task_indices, b_size=batch_size)
    observations = anchor_context[..., :algo.obs_dim]
    repeated_z = task_z.unsqueeze(1).expand(-1, batch_size, -1)
    if use_policy_relabel_data is None:
        use_policy_relabel_data = algo.consistency_use_policy_relabel_data
    if use_policy_relabel_data:
        policy_inputs = torch.cat([observations.reshape(-1, algo.obs_dim),
                                   repeated_z.reshape(-1, algo.latent_dim)], -1)
        actions = algo.agent.policy(count, batch_size, policy_inputs,
                                    reparameterize=True, return_log_prob=True)[0]
        actions = actions.reshape(count, batch_size, algo.action_dim)
    else:
        other_tasks = algo._sample_mismatched_task_indices(anchor_task_indices)
        other_context = algo.sample_context(other_tasks, b_size=batch_size)
        actions = other_context[..., algo.obs_dim:algo.obs_dim + algo.action_dim]
    return torch.cat([observations, actions, algo.context_decoder(observations, actions, repeated_z)], -1)


def rng_state():
    return torch.get_rng_state().clone(), np.random.get_state()


def set_rng_state(state):
    torch.set_rng_state(state[0])
    np.random.set_state(state[1])


def assert_same_rng(first, second):
    assert torch.equal(first[0], second[0])
    assert first[1][0] == second[1][0]
    np.testing.assert_array_equal(first[1][1], second[1][1])
    assert first[1][2:] == second[1][2:]


@pytest.mark.parametrize('dynamics', [False, True])
@pytest.mark.parametrize('policy_actions', [False, True])
@pytest.mark.parametrize('input_mode', [None, 'legacy'])
def test_legacy_context_is_bit_identical_and_ignores_real_dispatch_attribute(dynamics, policy_actions, input_mode):
    algo, _ = make_algorithm(dynamics)
    algo.consistency_use_policy_relabel_data = policy_actions
    # Selecting the real branch must not silently change default virtual calls.
    algo.real_consistency_input_mode = 'paired_replay'
    z = torch.tensor([[0.3, 0.1], [0.8, -0.2]])
    initial_state = rng_state()
    expected = legacy_context_reference(algo, z, 16)
    expected_rng = rng_state()
    set_rng_state(initial_state)
    actual = algo._build_consistency_context(z, 16, input_mode=input_mode)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert_same_rng(rng_state(), expected_rng)


@pytest.mark.parametrize('input_mode,policy_actions', [('policy', True), ('cross_task', False)])
def test_explicit_nonpaired_mode_overrides_old_boolean(input_mode, policy_actions):
    algo, _ = make_algorithm()
    algo.consistency_use_policy_relabel_data = not policy_actions
    z = torch.tensor([[0.3, 0.1], [0.8, -0.2]])
    initial_state = rng_state()
    expected = legacy_context_reference(algo, z, 16, [0, 1], policy_actions)
    expected_rng = rng_state()
    set_rng_state(initial_state)
    actual = algo._build_consistency_context(z, 16, [0, 1], input_mode=input_mode)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert_same_rng(rng_state(), expected_rng)


def test_zero_consistency_weight_preserves_diagnostics_and_rng_but_only_reconstruction_gradient():
    traces = []
    for coefficient in [0.0, 0.35714285714285715]:
        algo, sample_calls = make_algorithm(enabled=False)
        algo.consistency_loss_weight = coefficient
        algo.consistency_use_policy_relabel_data = False
        algo.real_consistency_input_mode = 'legacy'
        algo.agent.context_encoder.scale.data.fill_(0.6)
        context = algo.sample_context([0, 1], 16)
        z = algo.agent.context_encoder(context).mean(1)
        predicted = algo.context_decoder(context[..., :1], context[..., 1:2], z[:, None, :].expand(-1, 16, -1))
        reconstruction = algo.recon_loss_weight * F.mse_loss(predicted, context[..., 2:])
        expected_gradient = torch.autograd.grad(reconstruction, algo.agent.context_encoder.scale)[0]
        recorded_losses = []
        original = algo._compute_consistency_loss

        def record_loss(*args, **kwargs):
            result = original(*args, **kwargs)
            result.retain_grad()
            recorded_losses.append(result)
            return result

        algo._compute_consistency_loss = record_loss
        torch.manual_seed(91)
        np.random.seed(91)
        algo._take_step([0, 1], context)
        assert len(recorded_losses) == 1
        assert algo.loss['consistency_loss'] > 0
        assert algo.loss['consistency_loss'] == pytest.approx(algo.loss['real_consistency_loss'])
        if coefficient == 0:
            torch.testing.assert_close(algo.agent.context_encoder.scale.grad, expected_gradient)
            assert recorded_losses[0].grad is None
            assert algo.loss['encoder_total_loss'] == pytest.approx(algo.loss['context_loss'])
        else:
            assert recorded_losses[0].grad.abs().sum().item() > 0
        traces.append((copy.deepcopy(sample_calls), rng_state(), algo.loss['consistency_loss']))
    assert traces[0][0] == traces[1][0]
    assert_same_rng(traces[0][1], traces[1][1])
    assert traces[0][2] == traces[1][2]


@pytest.mark.parametrize('nonfinite', [float('nan'), float('inf')])
def test_disabled_consistency_nonfinite_diagnostic_cannot_poison_encoder_update(nonfinite):
    states = []
    gradients = []
    for corrupt in [False, True]:
        algo, _ = make_algorithm(enabled=False)
        algo.consistency_loss_weight = 0.0
        algo.agent.context_encoder.scale.data.fill_(0.6)
        original = algo._compute_consistency_loss
        calls = []

        def diagnostic_loss(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(result)
            return result * nonfinite if corrupt else result

        algo._compute_consistency_loss = diagnostic_loss
        algo._take_step([0, 1], algo.sample_context([0, 1], 16))
        assert len(calls) == 1  # Diagnostics still run even though their loss is disabled.
        assert np.isfinite(algo.loss['encoder_total_loss'])
        assert algo.loss['encoder_total_loss'] == pytest.approx(algo.loss['context_loss'])
        if corrupt:
            assert not np.isfinite(algo.loss['consistency_loss'])
        gradient = algo.agent.context_encoder.scale.grad.detach().clone()
        assert torch.isfinite(gradient).all()
        gradients.append(gradient)
        states.append(copy.deepcopy(algo.agent.context_encoder.state_dict()))
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)


@pytest.mark.parametrize('virtual_mode', ['global', 'semantic'])
def test_real_input_mode_does_not_change_virtual_consistency_source(virtual_mode):
    algo, _ = make_algorithm()
    algo.real_consistency_input_mode = 'paired_replay'
    algo.consistency_use_policy_relabel_data = False
    algo.virtual_task_generation_mode = virtual_mode
    algo.virtual_transition_loss_weight = 0.0
    algo.virtual_transition_final_loss_weight = 0.0
    if virtual_mode == 'global':
        algo._semantic_interpolator = None
        algo.M, algo.beta = 2, 1.0
    computed = []
    mismatched_calls = []
    original_compute = algo._compute_consistency_loss
    original_mismatch = algo._sample_mismatched_task_indices

    def compute(task_z, batch_size, **kwargs):
        computed.append((task_z.shape[0], kwargs))
        return original_compute(task_z, batch_size, **kwargs)

    def mismatched(indices):
        mismatched_calls.append(tuple(indices))
        return original_mismatch(indices)

    algo._compute_consistency_loss = compute
    algo._sample_mismatched_task_indices = mismatched
    algo._take_step([0, 1], algo.sample_context([0, 1], 16))
    assert len(computed) == 2
    real, virtual = computed
    assert real[0] == 2 and real[1]['input_mode'] == 'paired_replay'
    assert real[1].get('support_batch') is None
    assert virtual[0] == 3 and virtual[1].get('input_mode') in (None, 'legacy')
    if virtual_mode == 'semantic':
        assert virtual[1]['support_batch'] is not None
        assert not mismatched_calls
    else:
        assert virtual[1].get('support_batch') is None
        assert len(mismatched_calls) == 1 and len(mismatched_calls[0]) == 3
    assert algo.loss['num_virtual_tasks'] == 3
    assert algo.eval_statistics['real_consistency_paired_replay_itr_mean'] == 1
    assert np.isfinite(algo.loss['encoder_total_loss'])


@pytest.mark.parametrize('suffix,changed_key,changed_value', [
    ('real-no-cons', 'consistency_loss_weight', 0.0),
    ('real-paired-cons', 'real_consistency_input_mode', 'paired_replay'),
])
def test_ant_profiles_change_only_intended_single_factor(suffix, changed_key, changed_value):
    directory = ROOT / 'configs' / 'interpolation-diagnostics'
    baseline = json.loads((directory / 'ant-dir-real-only-matched.json').read_text(encoding='utf-8'))
    path = directory / ('ant-dir-' + suffix + '.json')
    variant = json.loads(path.read_text(encoding='utf-8'))
    expected = copy.deepcopy(baseline)
    expected['algo_params'][changed_key] = changed_value
    assert variant == expected
    namespace, calls = training_entrypoint()
    runner = CliRunner()
    experiment_name = '37_' + suffix.replace('-', '_')
    result = runner.invoke(namespace['main'], [
        str(directory / 'ant-dir-real-only-matched.json'),
        '--exp_name', experiment_name, '--gpu', '0'])
    assert result.exit_code == 0, result.output
    result = CliRunner().invoke(namespace['main'], [
        str(path), '--exp_name', experiment_name, '--gpu', '0'])
    assert result.exit_code == 0, result.output
    assert [seed for _, seed in calls] == [0, 1, 2, 3] * 2
    old_resolved = copy.deepcopy(calls[0][0])
    new_resolved = copy.deepcopy(calls[4][0])
    old_resolved.pop('config_path')
    new_resolved.pop('config_path')
    old_resolved['algo_params'][changed_key] = changed_value
    assert new_resolved == old_resolved
    for resolved, _ in calls[4:]:
        assert resolved['algo_params'][changed_key] == changed_value
        assert resolved['algo_params']['n_vt'] == 0
        assert resolved['algo_params']['relabel_data_ratio'] == 0.95
        assert resolved['algo_params']['embedding_batch_size'] == 512
        assert resolved['require_pretrained_context'] is True


def test_cli_real_mode_overrides_default_without_leaking_between_invocations():
    namespace, calls = training_entrypoint()
    original_default = copy.deepcopy(namespace['default_config'])
    runner = CliRunner()
    path = str(ROOT / 'configs/interpolation-diagnostics/ant-dir-real-only-matched.json')
    for mode in ['paired_replay', 'policy', 'cross_task', 'legacy']:
        result = runner.invoke(namespace['main'], [
            path, '--seed_list', '0', '--real_consistency_input_mode', mode])
        assert result.exit_code == 0, result.output
        assert calls[-1][0]['algo_params']['real_consistency_input_mode'] == mode
        result = runner.invoke(namespace['main'], [path, '--seed_list', '0'])
        assert result.exit_code == 0, result.output
        assert calls[-1][0]['algo_params']['real_consistency_input_mode'] == 'legacy'
    prior_calls = len(calls)
    for invalid in ['paired', 'bad-mode', 'True']:
        result = runner.invoke(namespace['main'], [
            path, '--real_consistency_input_mode', invalid])
        assert result.exit_code == 2
    assert len(calls) == prior_calls
    assert namespace['default_config'] == original_default


@pytest.mark.parametrize('invalid', ['', 'paired', 'POLICY', None, True, 1, {}])
def test_invalid_json_real_modes_fail_at_algorithm_configuration(invalid):
    algo = GENTLE.__new__(GENTLE)
    with pytest.raises(ValueError, match='real_consistency_input_mode'):
        algo._configure_real_consistency_input({'real_consistency_input_mode': invalid})


def test_legacy_global_training_retains_one_combined_real_virtual_call():
    algo, _ = make_algorithm()
    algo.real_consistency_input_mode = 'legacy'
    algo.virtual_task_generation_mode = 'global'
    algo._semantic_interpolator = None
    algo.M, algo.beta = 2, 1.0
    algo.virtual_transition_loss_weight = 0
    algo.virtual_transition_final_loss_weight = 0
    calls = []
    original = algo._compute_consistency_loss

    def compute(task_z, batch_size, **kwargs):
        calls.append((task_z.shape[0], kwargs.get('input_mode')))
        return original(task_z, batch_size, **kwargs)

    algo._compute_consistency_loss = compute
    algo._take_step([0, 1], algo.sample_context([0, 1], 16))
    assert calls == [(5, None)]
    assert algo.loss['real_consistency_paired_replay'] == 0


@pytest.mark.parametrize('suffix,paired,weight', [
    ('real-no-cons', 0, 0.0),
    ('real-paired-cons', 1, 0.35714285714285715),
])
def test_new_profiles_train_and_publish_the_expected_real_only_weights(suffix, paired, weight):
    namespace, calls = training_entrypoint()
    path = ROOT / 'configs/interpolation-diagnostics' / ('ant-dir-' + suffix + '.json')
    result = CliRunner().invoke(namespace['main'], [str(path), '--seed_list', '0'])
    assert result.exit_code == 0, result.output
    params = calls[0][0]['algo_params']
    algo, _ = make_algorithm(enabled=False)
    for key in ('real_consistency_input_mode', 'consistency_loss_weight',
                'consistency_use_policy_relabel_data', 'virtual_task_generation_mode',
                'virtual_transition_loss_weight', 'virtual_transition_final_loss_weight'):
        setattr(algo, key, params[key])
    algo._semantic_interpolator = None
    for iteration in [0, 499]:
        algo.itr = iteration
        algo._take_step([0, 1], algo.sample_context([0, 1], 16))
        assert algo.loss['num_virtual_tasks'] == 0
        assert algo.loss['num_virtual_transitions_added'] == 0
        assert algo.loss['virtual_transition_batch_size'] == 0
        assert algo.loss['virtual_transition_policy_q_batch_size'] == 0
        assert algo.loss['virtual_consistency_effective_weight'] == 0
        assert algo.loss['real_consistency_effective_weight'] == pytest.approx(weight)
        assert algo.eval_statistics['real_consistency_paired_replay_itr_mean'] == paired
        assert algo.eval_statistics['real_consistency_effective_weight_itr_mean'] == pytest.approx(weight)
        for key in ('qf_loss', 'encoder_total_loss', 'policy_total_loss'):
            assert np.isfinite(algo.loss[key])
