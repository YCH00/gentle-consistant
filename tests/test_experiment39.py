"""Exercise experiment 39's actual launch settings and training boundaries."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from configs.default import default_config
from test_interpolation_ablation import training_entrypoint
from test_semantic_integration import SemanticTaskInterpolator, make_algorithm


CONFIGS = Path(__file__).resolve().parents[1] / 'configs/interpolation-diagnostics'


def effective_profile(filename):
    namespace, calls = training_entrypoint()
    result = CliRunner().invoke(namespace['main'], [str(CONFIGS / filename), '--gpu', '0'])
    assert result.exit_code == 0, result.output
    assert [seed for _, seed in calls] == [0, 1, 2, 3]
    assert namespace['default_config'] == default_config
    return calls[0][0]


@pytest.mark.parametrize('filename,baseline,changes', [
    ('ant-dir-global-rl-only.json', 'ant-dir-semantic-path0-rl-only.json',
     dict(virtual_task_generation_mode='global')),
    ('ant-dir-semantic-path0-virtual-cons-decay.json', 'ant-dir-semantic-path0-virtual-cons.json',
     dict(virtual_consistency_weight_schedule='linear_decay',
          virtual_consistency_weight_decay_start_itr=150,
          virtual_consistency_weight_decay_end_itr=300, virtual_consistency_final_weight=0.0)),
])
def test_effective_launch_settings_match_completed_controls(filename, baseline, changes):
    new, old = effective_profile(filename), effective_profile(baseline)
    expected = copy.deepcopy(old)
    expected['algo_params'].update(changes)
    for variant in (new, expected):
        variant.pop('config_path')
    assert new == expected
    params = new['algo_params']
    assert params['n_vt'] == 4 and params['M'] == params['virtual_semantic_neighbors'] == 2
    assert params['batch_size'] == params['embedding_batch_size'] == 512
    assert params['real_consistency_loss_weight'] == params['consistency_loss_weight'] == 0
    assert params['virtual_semantic_path_steps'] == params['virtual_semantic_training_bank_size'] == 0
    assert new['require_pretrained_context'] is True


@pytest.mark.parametrize('filename', ['ant-dir-global-rl-only.json', 'ant-dir-semantic-path0-virtual-cons-decay.json'])
def test_production_updates_decay_virtual_terms_without_restoring_real_consistency(filename):
    params = effective_profile(filename)['algo_params']
    algo, _ = make_algorithm()
    # Use the actual algorithm/network update with a small synthetic replay;
    # retain small real/context batch sizes instead of requiring MuJoCo data.
    for key in ('n_vt', 'M', 'beta', 'virtual_task_generation_mode',
                'consistency_use_policy_relabel_data', 'virtual_transition_loss_weight',
                'virtual_transition_weight_schedule', 'virtual_transition_weight_decay_start_itr',
                'virtual_transition_weight_decay_end_itr', 'virtual_transition_final_loss_weight',
                'virtual_transition_use_cycle_weight', 'virtual_transition_use_recon_weight',
                'virtual_transition_use_semantic_weight', 'virtual_transition_train_policy_q',
                'virtual_transition_train_policy_bc'):
        setattr(algo, key, params[key])
    algo._configure_consistency_weights(params)
    algo._configure_virtual_consistency_schedule(params)
    algo._configure_real_consistency_input(params)
    options = {key: value for key, value in params.items() if key.startswith('virtual_semantic_')}
    algo._semantic_interpolator = (SemanticTaskInterpolator(algo.context_decoder, False, **options)
                                   if params['virtual_task_generation_mode'] == 'semantic' else None)
    for iteration in [0, 149, 150, 151, 225, 299, 300, 499]:
        algo.itr = iteration
        algo._take_step([0, 1], algo.sample_context([0, 1], 16))
        gamma = max(0.0, min(1.0, (300 - iteration) / 150))
        assert algo.loss['num_virtual_tasks'] == 4
        assert algo.loss['real_consistency_effective_weight'] == 0
        assert algo.loss['virtual_consistency_effective_weight'] == pytest.approx(
            params['virtual_consistency_loss_weight'] * gamma)
        assert algo.loss['virtual_transition_loss_weight_current'] == pytest.approx(0.5 * gamma)
        assert (algo.loss['virtual_transition_batch_size'] > 0) == (gamma > 0)
        assert algo.loss['virtual_transition_policy_bc_batch_size'] == 0
        for key in ('qf_loss', 'policy_total_loss', 'encoder_total_loss'):
            assert np.isfinite(algo.loss[key])
        if params['virtual_consistency_loss_weight'] == 0 or gamma == 0:
            assert algo.loss['consistency_encoder_loss'] == 0
            assert algo.loss['encoder_total_loss'] == algo.loss['context_loss']
        else:
            assert algo.loss['consistency_encoder_loss'] == pytest.approx(
                params['virtual_consistency_loss_weight'] * gamma * algo.loss['virtual_consistency_loss'])
        algo._num_steps += 1

