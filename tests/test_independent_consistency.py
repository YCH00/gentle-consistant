"""Check independent consistency gradients, historical behavior and launch profiles."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from configs.default import default_config
from test_interpolation_ablation import training_entrypoint
from test_semantic_integration import GENTLE, SemanticTaskInterpolator, make_algorithm


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs/interpolation-diagnostics'


@pytest.mark.parametrize('real_weight,virtual_weight', [
    (None, None), (0.1, None), (None, 0.2), (0.0, 0.2), (0.1, 0.0), (0.0, 0.0)])
@pytest.mark.parametrize('iteration,gamma', [(0, 1.0), (3, 0.5), (4, 0.0)])
def test_group_mean_coefficients_and_gradients(real_weight, virtual_weight, iteration, gamma):
    algo, _ = make_algorithm()
    algo._configure_consistency_weights(dict(
        consistency_loss_weight=0.5, real_consistency_loss_weight=real_weight,
        virtual_consistency_loss_weight=virtual_weight))
    algo._configure_virtual_consistency_schedule(dict(
        virtual_consistency_weight_schedule='linear_decay',
        virtual_consistency_weight_decay_start_itr=2,
        virtual_consistency_weight_decay_end_itr=4))
    algo.itr = iteration
    real = torch.arange(1., 4., requires_grad=True)
    virtual = torch.arange(4., 6., requires_grad=True)
    raw = algo._aggregate_consistency_losses(torch.cat([real, virtual]), 3)
    weighted = algo._weighted_consistency_loss(real, virtual, raw)
    expected_real = 0.5 * 3 / 5 if real_weight is None else real_weight
    expected_virtual = (0.5 * 2 / 5 if virtual_weight is None else virtual_weight) * gamma
    assert algo.loss['real_consistency_effective_weight'] == pytest.approx(expected_real)
    assert algo.loss['virtual_consistency_effective_weight'] == pytest.approx(expected_virtual)
    if expected_real == expected_virtual == 0:
        assert weighted is None
        return
    assert weighted.item() == pytest.approx(expected_real * 2 + expected_virtual * 4.5)
    weighted.backward()
    for tensor, coefficient in [(real, expected_real), (virtual, expected_virtual)]:
        if tensor.grad is None:
            assert coefficient == 0
        else:
            torch.testing.assert_close(tensor.grad, torch.full_like(tensor, coefficient / tensor.numel()))


@pytest.mark.parametrize('disabled_group', ['real', 'virtual'])
@pytest.mark.parametrize('nonfinite', [float('nan'), float('inf')])
@pytest.mark.parametrize('mode', ['semantic', 'global'])
def test_one_disabled_branch_cannot_poison_the_other_encoder_gradient(disabled_group, nonfinite, mode):
    states, gradients = [], []
    for corrupt in [False, True]:
        algo, _ = make_algorithm()
        algo._configure_consistency_weights(dict(
            consistency_loss_weight=0.0,
            real_consistency_loss_weight=0.0 if disabled_group == 'real' else 0.1,
            virtual_consistency_loss_weight=0.0 if disabled_group == 'virtual' else 0.2))
        algo.virtual_task_generation_mode = mode
        if mode == 'global':
            algo._semantic_interpolator = None
            algo.M, algo.beta = 2, 1.0
        algo.virtual_transition_loss_weight = algo.virtual_transition_final_loss_weight = 0.0
        if mode == 'semantic':
            # Use a production-validated cached support batch so the graph
            # isolation test is independent of later prototype rejection.
            z, support = algo._sample_virtual_task_embeddings(16, return_metadata=True)
            assert z is not None and support is not None
            algo._sample_virtual_task_embeddings = lambda *a, **kw: (z, support)
        algo.agent.context_encoder.scale.data.fill_(0.6)
        original = algo._compute_consistency_loss
        losses = []

        def compute(*args, **kwargs):
            result = original(*args, **kwargs)
            is_real = len(losses) == 0
            if corrupt and (is_real == (disabled_group == 'real')):
                result = result * nonfinite
            result.retain_grad()
            losses.append(result)
            return result

        algo._compute_consistency_loss = compute
        algo._take_step([0, 1], algo.sample_context([0, 1], 16))
        assert len(losses) == 2
        disabled, active = (losses[0], losses[1]) if disabled_group == 'real' else (losses[1], losses[0])
        assert disabled.grad is None  # No zero gradient is sent through a nonfinite branch.
        assert active.grad is not None and active.grad.abs().sum() > 0
        assert np.isfinite(algo.loss['consistency_encoder_loss'])
        assert np.isfinite(algo.loss['encoder_total_loss'])
        gradient = algo.agent.context_encoder.scale.grad.detach().clone()
        assert torch.isfinite(gradient).all()
        gradients.append(gradient)
        states.append(copy.deepcopy(algo.agent.context_encoder.state_dict()))
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('weight', [0.0, 0.5])
def test_unset_group_weights_retain_the_legacy_update_and_rng(enabled, weight):
    outcomes = []
    for legacy_reference in [False, True]:
        algo, calls = make_algorithm(enabled=enabled)
        algo.consistency_loss_weight = weight
        if legacy_reference:
            algo._weighted_consistency_loss = lambda real, virtual, raw: None if weight == 0 else weight * raw
        for _ in range(2):
            algo._take_step([0, 1], algo.sample_context([0, 1], 16))
            algo._num_steps += 1
        states = [copy.deepcopy(module.state_dict()) for module in (
            algo.agent.context_encoder, algo.agent.policy, algo.qf1, algo.qf2,
            algo.target_qf1, algo.target_qf2)]
        outcomes.append((states, calls, torch.get_rng_state(), np.random.get_state()))
    for state, reference in zip(outcomes[0][0], outcomes[1][0]):
        for key in state:
            torch.testing.assert_close(state[key], reference[key], rtol=0, atol=0)
    assert outcomes[0][1] == outcomes[1][1]
    assert torch.equal(outcomes[0][2], outcomes[1][2])
    np.testing.assert_equal(outcomes[0][3], outcomes[1][3])


@pytest.mark.parametrize('key', ['consistency_loss_weight', 'real_consistency_loss_weight',
                                 'virtual_consistency_loss_weight'])
@pytest.mark.parametrize('invalid', [-1, float('nan'), float('inf'), -float('inf'), True, '0.1'])
def test_invalid_coefficients_fail_configuration(key, invalid):
    algo = GENTLE.__new__(GENTLE)
    with pytest.raises(ValueError, match=key):
        algo._configure_consistency_weights({key: invalid})


def test_group_cli_overrides_accept_zero_and_do_not_modify_defaults():
    namespace, calls = training_entrypoint()
    original = copy.deepcopy(namespace['default_config'])
    runner = CliRunner()
    path = str(CONFIGS / 'ant-dir-real-no-cons.json')
    result = runner.invoke(namespace['main'], [path, '--seed_list', '0',
        '--real_consistency_loss_weight', '0', '--virtual_consistency_loss_weight', '0.2'])
    assert result.exit_code == 0, result.output
    assert calls[-1][0]['algo_params']['real_consistency_loss_weight'] == 0
    assert calls[-1][0]['algo_params']['virtual_consistency_loss_weight'] == 0.2
    result = runner.invoke(namespace['main'], [path, '--seed_list', '0'])
    assert result.exit_code == 0, result.output
    assert calls[-1][0]['algo_params']['real_consistency_loss_weight'] is None
    assert calls[-1][0]['algo_params']['virtual_consistency_loss_weight'] is None
    assert namespace['default_config'] == original
    before = len(calls)
    for key in ['real_consistency_loss_weight', 'virtual_consistency_loss_weight']:
        for invalid in ['-1', 'nan', 'inf', '-inf']:
            assert runner.invoke(namespace['main'], [path, '--' + key, invalid]).exit_code == 2
    assert len(calls) == before


def test_experiment38_profiles_are_matched_and_launch_four_seeds():
    baseline = json.loads((CONFIGS / 'ant-dir-real-only-matched.json').read_text(encoding='utf-8'))
    no_cons = json.loads((CONFIGS / 'ant-dir-real-no-cons.json').read_text(encoding='utf-8'))
    low = json.loads((CONFIGS / 'ant-dir-real-low-cons.json').read_text(encoding='utf-8'))
    expected = copy.deepcopy(baseline)
    expected['algo_params']['consistency_loss_weight'] = 0.1
    assert low == expected
    rl_only = json.loads((CONFIGS / 'ant-dir-semantic-path0-rl-only.json').read_text(encoding='utf-8'))
    expected = copy.deepcopy(no_cons)
    expected['algo_params'].update(n_vt=4, virtual_task_generation_mode='semantic',
        virtual_transition_loss_weight=0.5, real_consistency_loss_weight=0.0,
        virtual_consistency_loss_weight=0.0)
    assert rl_only == expected
    virtual_cons = json.loads((CONFIGS / 'ant-dir-semantic-path0-virtual-cons.json').read_text(encoding='utf-8'))
    expected = copy.deepcopy(rl_only)
    expected['algo_params']['virtual_consistency_loss_weight'] = 0.5 * 4 / 14
    assert virtual_cons == expected
    namespace, calls = training_entrypoint()
    for suffix in ['real-low-cons', 'semantic-path0-rl-only', 'semantic-path0-virtual-cons']:
        result = CliRunner().invoke(namespace['main'], [str(CONFIGS / ('ant-dir-' + suffix + '.json'))])
        assert result.exit_code == 0, result.output
    assert [seed for _, seed in calls] == [0, 1, 2, 3] * 3
    for variant, _ in calls:
        params = variant['algo_params']
        assert params['batch_size'] == params['embedding_batch_size'] == 512
        assert params['embedding_mini_batch_size'] == 512 and params['relabel_data_ratio'] == 0.95
        assert params['real_consistency_input_mode'] == 'legacy'
        assert params['virtual_semantic_path_steps'] == params['virtual_semantic_training_bank_size'] == 0
        assert params['virtual_transition_weight_decay_start_itr'] == 150
        assert params['virtual_transition_weight_decay_end_itr'] == 300
        assert variant['require_pretrained_context'] is True


@pytest.mark.parametrize('suffix,real_weight,virtual_weight', [
    ('real-low-cons', 0.1, 0.0), ('semantic-path0-rl-only', 0.0, 0.0),
    ('semantic-path0-virtual-cons', 0.0, 0.5 * 4 / 14)])
def test_new_profiles_train_at_schedule_boundaries_and_when_generation_skips(suffix, real_weight, virtual_weight):
    params = {**copy.deepcopy(default_config['algo_params']),
              **json.loads((CONFIGS / ('ant-dir-' + suffix + '.json')).read_text(encoding='utf-8'))['algo_params']}
    enabled = params['n_vt'] > 0
    algo, _ = make_algorithm(enabled=enabled)
    for key in ('consistency_loss_weight', 'real_consistency_input_mode',
                'consistency_use_policy_relabel_data', 'virtual_transition_loss_weight',
                'virtual_transition_weight_schedule', 'virtual_transition_weight_decay_start_itr',
                'virtual_transition_weight_decay_end_itr', 'virtual_transition_final_loss_weight',
                'virtual_task_generation_mode', 'virtual_transition_use_cycle_weight',
                'virtual_transition_use_recon_weight', 'virtual_transition_use_semantic_weight',
                'virtual_semantic_probe_batch_size'):
        setattr(algo, key, params[key])
    algo.n_vt = params['n_vt']
    algo._configure_consistency_weights(params)
    algo._configure_virtual_consistency_schedule(params)
    semantic_options = {key: value for key, value in params.items() if key.startswith('virtual_semantic_')}
    algo._semantic_interpolator = SemanticTaskInterpolator(algo.context_decoder, False, **semantic_options) if enabled else None
    for iteration in [0, 150, 225, 300, 499]:
        algo.itr = iteration
        algo._take_step([0, 1], algo.sample_context([0, 1], 16))
        assert algo.loss['num_virtual_tasks'] == (4 if enabled else 0)
        assert algo.loss['real_consistency_effective_weight'] == pytest.approx(real_weight)
        assert algo.loss['virtual_consistency_effective_weight'] == pytest.approx(virtual_weight)
        expected_rl = max(0.0, min(0.5, 0.5 * (300 - iteration) / 150)) if enabled else 0.0
        assert algo.loss['virtual_transition_loss_weight_current'] == pytest.approx(expected_rl)
        assert (algo.loss['virtual_transition_batch_size'] > 0) == (expected_rl > 0)
        for key in ('encoder_total_loss', 'qf_loss', 'policy_total_loss', 'consistency_encoder_loss'):
            assert np.isfinite(algo.loss[key]), key
        if real_weight == virtual_weight == 0:
            assert algo.loss['encoder_total_loss'] == algo.loss['context_loss']
        algo._num_steps += 1
    algo._sample_virtual_task_embeddings = lambda *a, **kw: (None, None)
    algo._take_step([0, 1], algo.sample_context([0, 1], 16))
    assert algo.loss['num_virtual_tasks'] == 0
    assert algo.loss['real_consistency_effective_weight'] == pytest.approx(real_weight)
    assert algo.loss['virtual_consistency_effective_weight'] == 0
