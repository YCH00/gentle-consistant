"""Check the resolved no-virtual-task control and its actual training weights."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from test_interpolation_ablation import ENVIRONMENTS, training_entrypoint
from test_semantic_integration import make_algorithm


ROOT = Path(__file__).resolve().parents[1]


def resolved_profiles(env):
    namespace, calls = training_entrypoint()
    for folder, suffix in [('interpolation-ablation', 'global'),
                           ('interpolation-diagnostics', 'real-only-matched')]:
        path = ROOT / 'configs' / folder / (env + '-' + suffix + '.json')
        result = CliRunner().invoke(namespace['main'], [
            str(path), '--exp_name', '36_real_only_matched', '--gpu', '0',
        ])
        assert result.exit_code == 0, result.output
        assert [seed for _, seed in calls[-4:]] == [0, 1, 2, 3]
    return copy.deepcopy(calls[0][0]), copy.deepcopy(calls[4][0])


@pytest.mark.parametrize('env', ENVIRONMENTS)
def test_real_only_control_changes_only_generation_and_matching_weights(env):
    old, new = resolved_profiles(env)
    p, q = old['algo_params'], new['algo_params']
    # The training loop selects meta_batch entries, with replacement if needed.
    # Real consistency is evaluated once per entry, not per unique task ID.
    real_count = p['meta_batch']
    assert real_count == 10 and p['n_vt'] == 4
    expected = p['consistency_loss_weight'] * real_count / (real_count + p['n_vt'])
    assert q['consistency_loss_weight'] == pytest.approx(expected)
    assert q['n_vt'] == 0
    assert q['virtual_transition_loss_weight'] == 0
    assert q['virtual_transition_final_loss_weight'] == 0
    assert q['virtual_consistency_weight_schedule'] == 'constant'
    assert new['require_pretrained_context'] is True
    for key in ('n_vt', 'consistency_loss_weight', 'virtual_transition_loss_weight',
                'virtual_transition_final_loss_weight', 'virtual_consistency_weight_schedule'):
        q[key] = p[key]
    old.pop('config_path')
    new.pop('config_path')
    assert old == new


@pytest.mark.parametrize('env', ENVIRONMENTS)
def test_real_only_control_preserves_real_consistency_gradients(env):
    old, new = resolved_profiles(env)
    old_params, new_params = old['algo_params'], new['algo_params']
    real_count, virtual_count = old_params['meta_batch'], old_params['n_vt']
    reference, _ = make_algorithm()
    control, _ = make_algorithm(enabled=False)
    reference.consistency_loss_weight = old_params['consistency_loss_weight']
    control.consistency_loss_weight = new_params['consistency_loss_weight']
    reference._configure_virtual_consistency_schedule(old_params)
    control._configure_virtual_consistency_schedule(new_params)
    original_losses = torch.arange(1., real_count + virtual_count + 1, requires_grad=True)
    control_losses = original_losses[:real_count].detach().clone().requires_grad_(True)
    reference_loss = reference.consistency_loss_weight * reference._aggregate_consistency_losses(
        original_losses, real_count)
    control_loss = control.consistency_loss_weight * control._aggregate_consistency_losses(
        control_losses, real_count)
    reference_loss.backward()
    control_loss.backward()
    torch.testing.assert_close(original_losses.grad[:real_count], control_losses.grad)
    assert reference.loss['real_consistency_effective_weight'] == pytest.approx(
        control.loss['real_consistency_effective_weight'])
    assert control.loss['virtual_consistency_effective_weight'] == 0


@pytest.mark.parametrize('env', ENVIRONMENTS)
def test_real_only_control_training_uses_no_virtual_samples(env):
    _, variant = resolved_profiles(env)
    params = variant['algo_params']
    dynamics = params['use_next_obs_in_context']
    algo, calls = make_algorithm(dynamics=dynamics, enabled=False)
    # Keep the small synthetic networks/data; exercise all relevant profile
    # switches with production dispatch, loss construction and optimizer steps.
    for key in ('n_vt', 'meta_batch', 'consistency_loss_weight', 'virtual_task_generation_mode',
                'consistency_use_policy_relabel_data', 'virtual_transition_loss_weight',
                'virtual_transition_final_loss_weight', 'virtual_transition_weight_schedule',
                'virtual_transition_weight_decay_start_itr', 'virtual_transition_weight_decay_end_itr',
                'virtual_transition_use_recon_weight', 'virtual_transition_use_cycle_weight',
                'virtual_transition_train_policy_q', 'virtual_transition_train_policy_bc'):
        setattr(algo, key, params[key])
    algo._configure_virtual_consistency_schedule(params)
    initial_calls = len(calls)
    assert algo._sample_virtual_task_embeddings(16, return_metadata=True) == (None, None)
    assert len(calls) == initial_calls

    def forbidden(*args, **kwargs):
        raise AssertionError('the matched real-only control must not use virtual RL data')

    algo.add_virtual_transitions_to_buffer = forbidden
    algo.sample_virtual_sac = forbidden
    for iteration in (0, 499):
        algo.itr = iteration
        # Repeated task IDs also confirm real_count uses the meta-batch entries.
        indices = [0, 1] * (params['meta_batch'] // 2)
        result = algo._take_step(indices, algo.sample_context(indices, 16))
        assert result[0].shape == (params['meta_batch'], 2)
        assert algo.loss['num_virtual_tasks'] == 0
        assert algo.loss['num_virtual_transitions_added'] == 0
        assert algo.loss['virtual_transition_batch_size'] == 0
        assert algo.loss['virtual_transition_policy_q_batch_size'] == 0
        assert algo.loss['virtual_transition_policy_bc_batch_size'] == 0
        assert algo.loss['virtual_transition_loss_weight_current'] == 0
        assert algo.loss['virtual_consistency_fraction'] == 0
        assert algo.loss['real_consistency_effective_weight'] == pytest.approx(
            params['consistency_loss_weight'])
        assert algo.loss['consistency_loss'] == pytest.approx(algo.loss['real_consistency_loss'])
        assert algo.virtual_transition_buffer.size() == 0
        for key in ('qf_loss', 'encoder_total_loss', 'policy_total_loss'):
            assert np.isfinite(algo.loss[key])
