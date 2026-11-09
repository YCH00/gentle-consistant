"""Validate, smoke-test, or train GENTLE with mixed R/M/E data.

Uses the repository's existing algorithm and native evaluation settings.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from configs.default import default_config
from rlkit.data_management.offline_dataset import training_paths, evaluation_paths

ROOT = Path(__file__).resolve().parent
ENVIRONMENTS = ('point-robot', 'ant-dir', 'cheetah-vel', 'cheetah-dir',
                'hopper-rand-params', 'walker-rand-params')


def merge(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('validate', 'smoke', 'train'))
    parser.add_argument('--env', required=True, choices=ENVIRONMENTS)
    parser.add_argument('--project-root', type=Path, default=ROOT.parent)
    parser.add_argument('--config', type=Path, help='Optional base configuration; algorithm hyperparameters are preserved')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, action='append', help='Repeat for multiple seeds; train defaults to 0,1,2,3')
    parser.add_argument('--context-pretraining', choices=('required', 'disabled'), default='required')
    args = parser.parse_args()
    config_path = args.config or ROOT / 'configs' / (args.env + '-mixed-rme-v3.json')
    variant = merge(copy.deepcopy(default_config), json.loads(config_path.read_text(encoding='utf-8')))
    if variant['env_name'] != args.env:
        parser.error('Config environment differs from --env')
    project = args.project_root.resolve()
    params = variant['algo_params']
    params.update(train_data_format='mixed_rme_v3', dataset_tag='mixed_rme_v3',
                  data_dir=str(ROOT / 'data_mixed_rme_v3' / args.env),
                  train_artifacts_dir=str(project / 'shared_protocol/shared-native-v3' / args.env),
                  eval_data_dir=str(ROOT / 'data' / args.env),
                  eval_artifacts_dir=str(project / 'shared_protocol/shared-main-v1' / args.env))
    variant['context_pretraining'] = args.context_pretraining
    if not params.get('load_eval_buffer'):
        parser.error('Mixed experiment requires load_eval_buffer=true')
    tasks = list(range(variant['env_params']['n_tasks']))
    train = training_paths(params, tasks[:variant['n_train_tasks']])
    evaluation = evaluation_paths(params, tasks[-variant['n_eval_tasks']:])
    print('Validation passed: train=%d files; historical evaluation=%d files' % (len(train), len(evaluation)), flush=True)
    if args.stage == 'validate':
        return
    seeds = args.seed or ([0] if args.stage == 'smoke' else [0, 1, 2, 3])
    if args.stage == 'smoke':
        params['dataset_tag'] = 'mixed_rme_v3_smoke'
        params.update(num_iterations=2, num_train_steps_per_itr=1)
        variant.update(num_iters=2, decoder_iter=1, dynamics_max_epochs=1)
    run_dir = ROOT / 'output' / params['dataset_tag'] / args.env
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = run_dir / 'config.json'
    resolved.write_text(json.dumps(variant, indent=2), encoding='utf-8')
    stages = ['pretrain_dynamics.py']
    if args.context_pretraining == 'required':
        stages.append('pretrain_encoder_decoder.py')
    stages.append('train_gentle.py')
    for seed in seeds:
        for stage in stages:
            command = [sys.executable, str(ROOT / stage), str(resolved), '--gpu', str(args.gpu), '--seed_list', str(seed)]
            if stage == 'train_gentle.py':
                command += ['--output_prefix', params['dataset_tag'] + '_']
            print('Running: ' + ' '.join(command), flush=True)
            subprocess.run(command, cwd=str(ROOT), check=True)
    print('Completed %s for %s, seeds=%s' % (args.stage, args.env, seeds), flush=True)


if __name__ == '__main__':
    main()
