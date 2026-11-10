"""Validate, smoke-test, or train GENTLE with published mixed R/M/E data."""
import argparse
from copy import deepcopy
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback

from configs.default import default_config
from rlkit.data_management.offline_sources import (
    read_json, sha256, require, validate_bundle, dynamics_path,
)

ROOT = Path(__file__).resolve().parent
ENVIRONMENTS = ('point-robot', 'ant-dir', 'cheetah-vel', 'cheetah-dir',
                'hopper-rand-params', 'walker-rand-params')


def merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            merge(target.setdefault(key, {}), value)
        else:
            target[key] = value


def configure(args):
    variant = deepcopy(default_config)
    merge(variant, read_json(ROOT / 'configs' / (args.env + '.json')))
    variant.update(experiment_name='0_gentle_base_v3' + ('_smoke' if args.stage == 'smoke' else ''),
                   log_group='mixed_rme_v3_gentle', output_prefix='')
    variant['util_params'].update(gpu_id=args.gpu, base_log_dir=str(ROOT / 'logs'))
    params = variant['algo_params']
    params.update(data_dir=str(Path(args.raw_root).resolve() / args.env),
                  eval_data_dir=str(Path(args.eval_raw_root).resolve() / args.env),
                  mixed_artifacts_dir=str(Path(args.artifacts_root).resolve() / args.env),
                  eval_artifacts_dir=str(Path(args.eval_artifacts_root).resolve() / args.env))
    if args.stage == 'smoke':
        variant['dynamics_max_epochs'] = 1
        params.update(num_iterations=1, num_train_steps_per_itr=2, num_evals=1)
    return variant


def validate(variant):
    p = variant['algo_params']
    n = variant['n_train_tasks']
    total = variant['env_params']['n_tasks']
    train_tasks = list(range(n))
    eval_tasks = list(range(total - variant['n_eval_tasks'], total))
    train, train_hashes = validate_bundle(p['data_dir'], p['mixed_artifacts_dir'], train_tasks, 'train', True)
    evaluation, eval_hashes = validate_bundle(p['eval_data_dir'], p['eval_artifacts_dir'], eval_tasks, 'eval')
    require(train_hashes['task_manifest.json'] == eval_hashes['task_manifest.json'], 'Training/evaluation task identity mismatch')
    provenance = read_json(Path(p['mixed_artifacts_dir']) / 'preparation_provenance.json')
    require(provenance['old_dataset_manifest_sha256'] == eval_hashes['dataset_manifest.json'], 'Historical dataset identity mismatch')
    manifest = read_json(Path(p['mixed_artifacts_dir']) / 'task_manifest.json')
    require(manifest['environment'] == variant['env_name'], 'Environment mismatch')
    require(manifest['horizon'] == p['max_path_length'], 'Horizon mismatch')
    expected = 100 if variant['env_name'] in ('point-robot', 'cheetah-dir') else 50
    require(all(sum(t == task for _, t in train) == expected for task in train_tasks), 'Mixed trajectory quota mismatch')
    dataset = read_json(Path(p['mixed_artifacts_dir']) / 'dataset_manifest.json')
    quota = (34, 33, 33) if expected == 100 else (17, 17, 16)
    for task in train_tasks:
        entries = [e for e in dataset['files'] if e['task_id'] == task]
        require(tuple(sum(e['behavior'] == b for e in entries) for b in ('random', 'medium', 'expert')) == quota,
                'Random/Medium/Expert quota mismatch')
        require(dataset['task_rows'][str(task)] <= p['replay_buffer_size'], 'Replay buffer would truncate task')
    variant['data_receipt'] = dict(training=train_hashes, evaluation=eval_hashes,
                                 training_trajectories=len(train), evaluation_trajectories=len(evaluation),
                                 evaluation_protocol='native GENTLE (not frozen given-context bank)')
    print(json.dumps(variant['data_receipt'], indent=2), flush=True)


def worker(variant, seed, stage):
    # Each spawned process owns its CUDA context, logger and seed-specific dynamics.
    log_dir = (ROOT / 'logs' / variant['env_name'] / variant['log_group'] /
               ('seed%d' % seed) / variant['experiment_name'])
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / (stage + '_console.log')).open('w', buffering=1) as output:
        import sys
        sys.stdout = sys.stderr = output
        try:
            import torch
            torch.set_num_threads(1)
            variant['seed'] = seed
            (log_dir / 'data_receipt.json').write_text(json.dumps(variant['data_receipt'], indent=2), encoding='utf-8')
            if stage in ('all', 'pretrain', 'smoke'):
                from pretrain_dynamics import experiment
                experiment(deepcopy(variant), seed)
            if stage in ('all', 'train', 'smoke'):
                from train_gentle import experiment
                experiment(deepcopy(variant), seed)
        except BaseException:
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('validate', 'smoke', 'pretrain', 'train', 'all'))
    parser.add_argument('--env', required=True, choices=ENVIRONMENTS)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--raw-root', default=str(ROOT / 'data_mixed_rme_v3'))
    parser.add_argument('--eval-raw-root', default=str(ROOT / 'data'))
    parser.add_argument('--artifacts-root', default=str(ROOT.parent / 'shared_protocol/shared-native-v3'))
    parser.add_argument('--eval-artifacts-root', default=str(ROOT.parent / 'shared_protocol/shared-main-v1'))
    args = parser.parse_args()
    require(len(args.seeds) == len(set(args.seeds)), 'Seeds must be unique')
    variant = configure(args)
    validate(variant)
    if args.stage == 'validate':
        return
    # Refuse accidental overwrites; smoke has separate model and log directories.
    for seed in args.seeds:
        log = ROOT / 'logs' / args.env / variant['log_group'] / ('seed%d' % seed) / variant['experiment_name']
        if args.stage != 'pretrain':
            require(not (log / 'variant.json').exists(), 'Training output already exists: ' + str(log))
        if args.stage != 'train':
            require(not dynamics_path(variant, seed).exists(), 'Dynamics output already exists: ' + str(dynamics_path(variant, seed)))
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    ctx = mp.get_context('spawn')
    processes = [ctx.Process(target=worker, args=(variant, seed, args.stage)) for seed in args.seeds]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failures = [seed for seed, process in zip(args.seeds, processes) if process.exitcode != 0]
    require(not failures, 'Failed seeds (see per-seed console logs): ' + str(failures))


if __name__ == '__main__':
    main()
