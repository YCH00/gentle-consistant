"""Validate resolved launch configurations without importing Gym/MuJoCo."""

import ast
import copy
import hashlib
import json
import os
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from configs.default import default_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs/interpolation-ablation'
ENVIRONMENTS = ['ant-dir', 'cheetah-vel', 'point-robot',
                'hopper-rand-params', 'walker-rand-params']
ARMS = [('global', 'global', 0), ('semantic-path0', 'semantic', 0),
        ('semantic-path8', 'semantic', 8)]


def training_entrypoint():
    calls = []

    def experiment(variant, seed=None):
        calls.append((copy.deepcopy(variant), seed))

    class Pool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starmap(self, fn, arguments):
            for args in arguments:
                fn(*args)

        def close(self):
            pass

        def join(self):
            pass

    source = ROOT / 'train_gentle.py'
    names = {'deep_update_dict', 'main', '_resolve_weights_dir', '_find_weight_file',
             '_load_pretrained_context_models', '_checkpoint_sha256'}
    functions = [node for node in ast.parse(source.read_text(encoding='utf-8')).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(copy=copy, hashlib=hashlib, json=json, os=os, click=click,
                     Path=Path, product=product, mp=SimpleNamespace(Pool=Pool),
                     default_config=copy.deepcopy(default_config), experiment=experiment,
                     __file__=str(source))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace, calls


@pytest.mark.parametrize('env', ENVIRONMENTS)
def test_three_arms_match_after_actual_cli_and_default_merge(env):
    namespace, calls = training_entrypoint()
    pristine_default = copy.deepcopy(namespace['default_config'])
    variants = []
    for arm, mode, steps in ARMS:
        result = CliRunner().invoke(namespace['main'], [
            str(CONFIGS / f'{env}-{arm}.json'), '--gpu', '1', '--exp_name', arm,
        ])
        assert result.exit_code == 0, result.output
        assert [seed for _, seed in calls[-4:]] == [0, 1, 2, 3]
        variant = calls[-1][0]
        assert variant['seed_list'] == [0, 1, 2, 3]
        assert variant['util_params']['gpu_id'] == 1
        assert variant['exp_name'] == arm
        assert variant['require_pretrained_context'] is True
        assert Path(variant['config_path']).is_file()
        params = variant['algo_params']
        assert params['n_vt'] == 4
        assert params['virtual_task_generation_mode'] == mode
        assert params['virtual_semantic_path_steps'] == steps
        assert params['virtual_transition_weight_schedule'] == 'linear_decay'
        assert params['virtual_transition_final_loss_weight'] == 0
        assert params['virtual_transition_train_policy_q'] is True
        for key in ('virtual_transition_use_recon_weight', 'virtual_transition_use_cycle_weight',
                    'virtual_transition_use_semantic_weight', 'virtual_transition_train_policy_bc',
                    'virtual_transition_use_policy_actions'):
            assert params[key] is False
        # Compare every effective hyperparameter, not just a shortlist.
        normalized = copy.deepcopy(variant)
        for key in ('exp_name', 'config_path'):
            del normalized[key]
        for key in ('virtual_task_generation_mode', 'virtual_semantic_path_steps'):
            del normalized['algo_params'][key]
        variants.append(normalized)
    assert variants[0] == variants[1] == variants[2]
    assert namespace['default_config'] == pristine_default


def test_discrete_control_does_not_claim_three_interpolation_arms():
    namespace, calls = training_entrypoint()
    result = CliRunner().invoke(namespace['main'], [
        str(CONFIGS / 'cheetah-dir-control.json'), '--seed_list', '2',
    ])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and calls[0][1] == 2
    params = calls[0][0]['algo_params']
    assert params['n_vt'] == 0 and params['virtual_transition_loss_weight'] == 0
    assert len(list(CONFIGS.glob('cheetah-dir-*.json'))) == 1


def test_cli_quality_override_is_explicit_and_does_not_leak_between_invocations():
    namespace, calls = training_entrypoint()
    path = str(CONFIGS / 'ant-dir-semantic-path8.json')
    runner = CliRunner()
    result = runner.invoke(namespace['main'], [path, '--virtual_transition_use_semantic_weight', 'true',
                                              '--n_vt', '9', '--seed_list', '0'])
    assert result.exit_code == 0, result.output
    assert calls[-1][0]['algo_params']['virtual_transition_use_semantic_weight'] is True
    assert calls[-1][0]['algo_params']['n_vt'] == 9
    result = runner.invoke(namespace['main'], [path, '--seed_list', '0'])
    assert result.exit_code == 0, result.output
    assert calls[-1][0]['algo_params']['virtual_transition_use_semantic_weight'] is False
    assert calls[-1][0]['algo_params']['n_vt'] == 4
    assert runner.invoke(namespace['main'], [path, '--n_vt', '-1']).exit_code == 2
    assert runner.invoke(namespace['main'], [path, '--virtual_semantic_path_steps', '-1']).exit_code == 2


@pytest.mark.parametrize('arm,mode,steps', ARMS)
def test_all_arms_require_pretrained_context_not_random_decoder(tmp_path, arm, mode, steps):
    namespace, _ = training_entrypoint()
    variant = json.loads((CONFIGS / f'ant-dir-{arm}.json').read_text())
    namespace['_resolve_weights_dir'] = lambda variant, seed: tmp_path
    with pytest.raises(FileNotFoundError, match='requires pretrained'):
        namespace['_load_pretrained_context_models'](None, None, variant, 0)


def test_pretrained_paths_and_hashes_record_actual_files(tmp_path):
    namespace, _ = training_entrypoint()
    encoder = tmp_path / 'context_encoder.pth'
    decoder = tmp_path / 'context_decoder.pth'
    encoder.write_bytes(b'encoder test fixture')
    decoder.write_bytes(b'decoder test fixture')
    loaded = []
    model = SimpleNamespace(load=lambda path: loaded.append(path))
    variant = dict(path_to_weights=str(tmp_path), algo_params={})
    namespace['_load_pretrained_context_models'](model, model, variant, 1)
    assert loaded == [encoder, decoder]
    metadata = variant['pretrained_context']
    assert metadata['encoder_path'] == str(encoder.resolve())
    assert metadata['encoder_sha256'] == hashlib.sha256(encoder.read_bytes()).hexdigest()
    assert metadata['decoder_sha256'] == hashlib.sha256(decoder.read_bytes()).hexdigest()


def test_saved_run_metadata_can_be_reloaded_by_embedding_exporter(tmp_path):
    namespace, calls = training_entrypoint()
    result = CliRunner().invoke(namespace['main'], [
        str(CONFIGS / 'ant-dir-semantic-path8.json'), '--seed_list', '0',
    ])
    assert result.exit_code == 0, result.output
    saved = calls[0][0]
    saved['seed'] = 0
    saved['pretrained_context'] = dict(encoder_path='/weights/encoder.pth',
                                     decoder_path='/weights/decoder.pth',
                                     encoder_sha256='abc', decoder_sha256='def')
    (tmp_path / 'variant.json').write_text(json.dumps(saved), encoding='utf-8')
    # Exercise the exporter's actual merge helper, which expects nested defaults.
    for filename, name in [('pretrain_encoder_decoder.py', 'deep_update_dict'),
                           ('export_virtual_embeddings.py', 'load_variant')]:
        source = ROOT / filename
        fn = next(node for node in ast.parse(source.read_text(encoding='utf-8')).body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
    loaded = namespace['load_variant'](tmp_path)
    assert loaded == saved
