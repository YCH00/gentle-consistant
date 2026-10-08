"""Validate passive, per-iteration critic observations on the training path."""

import copy

import numpy as np
import pytest
import torch

from test_semantic_integration import GENTLE, make_algorithm


def test_critic_distributions_keep_real_and_virtual_rows_separate():
    algo, _ = make_algorithm()
    q1 = torch.tensor([[1.], [3.], [100.]], requires_grad=True)
    q2 = torch.tensor([[2.], [5.], [200.]], requires_grad=True)
    target = torch.tensor([[4.], [8.], [30.]])
    bootstrap = torch.tensor([[7.], [9.], [40.]])
    algo._record_critic_diagnostics(q1, q2, target, bootstrap, 2)
    assert algo.loss['real_q1_pred_mean'] == 2
    assert algo.loss['real_q1_pred_std'] == 1
    assert algo.loss['real_q1_pred_min'] == 1
    assert algo.loss['real_q1_pred_max'] == 3
    assert algo.loss['real_q_target_mean'] == 6
    assert algo.loss['real_q_target_std'] == 2
    assert algo.loss['real_q_bootstrap_mean'] == 8
    assert algo.loss['real_td_abs_mean'] == 3.25
    assert algo.loss['real_td_abs_max'] == 5
    assert algo.loss['virtual_q1_pred_mean'] == 100
    assert algo.loss['virtual_q1_pred_std'] == 0  # Population std is finite for one row.
    assert algo.loss['virtual_td_abs_mean'] == 120
    assert algo.loss['virtual_td_abs_max'] == 170
    assert q1.grad is q2.grad is None
    algo._record_critic_diagnostics(q1[:2], q2[:2], target[:2], bootstrap[:2], 2)
    assert all(value == 0 for key, value in algo.loss.items() if key.startswith('virtual_'))


def test_gradient_norm_observes_all_gradients_without_changing_them_or_overflowing():
    module = torch.nn.Linear(2, 1)
    module.weight.grad = torch.tensor([[3., 4.]])
    module.bias.grad = torch.tensor([12.])
    before = [p.grad.clone() for p in module.parameters()]
    assert GENTLE._gradient_norm(module) == 13
    for parameter, original in zip(module.parameters(), before):
        torch.testing.assert_close(parameter.grad, original, rtol=0, atol=0)
    module.weight.grad.fill_(1e30)
    module.bias.grad = None
    assert GENTLE._gradient_norm(module) == pytest.approx(np.sqrt(2) * 1e30, rel=1e-6)
    module.weight.grad = None
    assert GENTLE._gradient_norm(module) == 0


def test_training_reports_bellman_target_separately_from_bootstrap_and_actor_normalization():
    algo, _ = make_algorithm(enabled=False)
    algo.reward_scale, algo.discount = 2.0, 0.75
    batch, target_outputs, behavior_outputs = [], [], []
    original_sample = algo.sample_sac

    def sample(indices):
        result = original_sample(indices)
        batch.append(result)
        return result

    algo.sample_sac = sample
    handles = [
        algo.target_qf1.register_forward_hook(lambda module, args, output: target_outputs.append(output.detach().clone())),
        algo.target_qf2.register_forward_hook(lambda module, args, output: target_outputs.append(output.detach().clone())),
        algo.qf1.register_forward_hook(lambda module, args, output: behavior_outputs.append(output.detach().clone())),
    ]
    algo._take_step([0, 1], algo.sample_context([0, 1], 16))
    for handle in handles:
        handle.remove()
    bootstrap = torch.minimum(*target_outputs)
    rewards = batch[0][2].reshape(-1, 1)
    terms = batch[0][4].reshape(-1, 1)
    target = 2 * rewards + (1 - terms) * 0.75 * bootstrap
    assert algo.loss['real_q_bootstrap_mean'] == pytest.approx(bootstrap.mean().item())
    assert algo.loss['real_q_target_mean'] == pytest.approx(target.mean().item())
    assert algo.loss['real_q_target_std'] == pytest.approx(target.std(unbiased=False).item())
    assert algo.loss['real_q1_pred_mean'] == pytest.approx(behavior_outputs[0].mean().item())
    assert algo.loss['q_target'] == pytest.approx(target.mean().item())
    assert algo.loss['qf1_grad_norm'] == pytest.approx(GENTLE._gradient_norm(algo.qf1))
    assert algo.loss['qf2_grad_norm'] == pytest.approx(GENTLE._gradient_norm(algo.qf2))
    assert algo.loss['policy_q_lambda'] == pytest.approx(
        algo.bc_weight / max(1e-6, algo.loss['policy_q_abs_mean']), rel=1e-6)
    assert algo.loss['policy_q_abs_mean'] == algo.loss['real_policy_q_abs_mean']
    for key in ('q_target_itr_mean', 'real_q_target_mean_itr_mean',
                'real_q_bootstrap_mean_itr_mean', 'qf1_grad_norm_itr_mean',
                'policy_q_lambda_itr_mean'):
        assert key in algo.eval_statistics and np.isfinite(algo.eval_statistics[key])


def test_iteration_means_and_extremes_include_every_call_and_reset():
    algo, _ = make_algorithm()
    snapshots = []
    for _ in range(3):
        algo._take_step([0, 1], algo.sample_context([0, 1], 16))
        snapshots.append(copy.deepcopy(algo.loss))
        algo._num_steps += 1
    for key in ('q_target', 'qf_loss', 'qf1_grad_norm', 'policy_q_lambda',
                'real_q1_pred_mean', 'virtual_q_target_std', 'consistency_encoder_loss'):
        assert algo.eval_statistics[key + '_itr_mean'] == pytest.approx(np.mean([s[key] for s in snapshots]))
    for key in ('qf_loss', 'qf1_grad_norm', 'policy_q_lambda', 'real_q_target_max', 'virtual_td_abs_max'):
        assert algo.eval_statistics[key + '_itr_max'] == max(s[key] for s in snapshots)
    for key in ('real_q1_pred_min', 'virtual_q_target_min', 'policy_q_lambda'):
        assert algo.eval_statistics[key + '_itr_min'] == min(s[key] for s in snapshots)
    schema = {key for key in algo.eval_statistics if key.startswith(('real_q', 'virtual_q', 'qf1_grad', 'qf2_grad'))}
    algo.itr += 1
    algo.eval_statistics = None
    algo.virtual_transition_loss_weight = algo.virtual_transition_final_loss_weight = 0.0
    algo._take_step([0, 1], algo.sample_context([0, 1], 16))
    assert algo.eval_statistics['virtual_diagnostic_training_calls'] == 1
    assert schema == {key for key in algo.eval_statistics if key.startswith(('real_q', 'virtual_q', 'qf1_grad', 'qf2_grad'))}
    for key in ('qf_loss', 'qf1_grad_norm', 'real_q_target_max'):
        assert algo.eval_statistics[key + '_itr_mean'] == algo.loss[key]
        assert algo.eval_statistics[key + '_itr_max'] == algo.loss[key]
    assert algo.eval_statistics['virtual_q_target_max_itr_max'] == 0


@pytest.mark.parametrize('enabled', [False, True])
def test_critic_observation_does_not_change_updates_or_random_state(enabled):
    outcomes = []
    for diagnostics in [False, True]:
        algo, calls = make_algorithm(enabled=enabled)
        if not diagnostics:
            algo._record_critic_diagnostics = lambda *args: None
        for _ in range(2):
            algo._take_step([0, 1], algo.sample_context([0, 1], 16))
            algo._num_steps += 1
        modules = [algo.agent.context_encoder, algo.agent.policy, algo.agent.target_policy,
                   algo.qf1, algo.qf2, algo.target_qf1, algo.target_qf2]
        outcomes.append(([copy.deepcopy(module.state_dict()) for module in modules],
                         copy.deepcopy(calls), torch.get_rng_state(), np.random.get_state()))
    for state, reference in zip(outcomes[0][0], outcomes[1][0]):
        for key in state:
            torch.testing.assert_close(state[key], reference[key], rtol=0, atol=0)
    assert outcomes[0][1] == outcomes[1][1]
    assert torch.equal(outcomes[0][2], outcomes[1][2])
    np.testing.assert_equal(outcomes[0][3], outcomes[1][3])


def test_nonfinite_diagnostic_is_not_hidden_by_iteration_extremes():
    algo, _ = make_algorithm(enabled=False)
    algo._take_step([0, 1], algo.sample_context([0, 1], 16))
    algo.loss['qf_loss'] = float('nan')
    algo.loss['real_q_target_min'] = float('nan')
    algo._record_virtual_task_diagnostics()
    assert np.isnan(algo.eval_statistics['qf_loss_itr_mean'])
    assert np.isnan(algo.eval_statistics['qf_loss_itr_max'])
    assert np.isnan(algo.eval_statistics['real_q_target_min_itr_min'])
