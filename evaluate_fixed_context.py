"""Re-evaluate saved GENTLE checkpoints with identical, non-growing contexts.

Environment imports are intentionally deferred until after argument parsing.
This script never constructs a training algorithm or writes into a run folder.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


FIELDS = ('observations', 'actions', 'rewards', 'next_observations', 'terminals')
MODEL_NAMES = ('context_encoder', 'context_decoder', 'policy', 'qf1', 'qf2')


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', required=True, type=Path)
    parser.add_argument('--epochs', nargs='+', type=int, default=[200, 400, 499])
    parser.add_argument('--collector-epochs', nargs='+', type=int, default=[200, 499])
    parser.add_argument('--collector-log-dir', type=Path, help='Optional shared reference run for online context collection.')
    parser.add_argument('--offline-only', action='store_true', help='Skip online context collection.')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--bank-dir', type=Path, help='Reusable context bank; defaults to output-dir/context_bank.')
    parser.add_argument('--data-dir', type=Path, help='Override the offline data location, not the file selection.')
    parser.add_argument('--split', choices=['test', 'train'], default='test')
    parser.add_argument('--context-size', type=positive_int, default=200)
    parser.add_argument('--context-repeats', type=positive_int, default=3)
    parser.add_argument('--probe-size', type=positive_int, default=256)
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--bank-seed', type=int, default=3600)
    parser.add_argument('--gpu', type=int, default=None, help='Omit for CPU; a requested unavailable GPU is an error.')
    parser.add_argument('--save-trajectories', action='store_true')
    args = parser.parse_args(argv)
    for name in ('epochs', 'collector_epochs', 'eval_seeds'):
        values = getattr(args, name)
        if any(v < 0 for v in values) or len(set(values)) != len(values):
            parser.error('--' + name.replace('_', '-') + ' requires distinct nonnegative integers')
    if any(seed >= 2**32 for seed in args.eval_seeds):
        parser.error('--eval-seeds values must be below 2**32')
    if args.bank_seed < 0 or (args.gpu is not None and args.gpu < 0):
        parser.error('seeds and GPU ids must be nonnegative')
    args.epochs.sort()
    args.collector_epochs.sort()
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def merge_dict(source, target):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_dict(value, target[key])
        else:
            target[key] = copy.deepcopy(value)
    return target


def load_variant(log_dir):
    from configs.default import default_config
    path = Path(log_dir) / 'variant.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    variant = merge_dict(saved, copy.deepcopy(default_config))
    if (not isinstance(saved.get('seed'), int) or isinstance(saved['seed'], bool)
            or not 0 <= saved['seed'] < 2**32):
        raise ValueError('variant.json must record the actual integer training seed')
    if variant['algo_params']['sparse_rewards']:
        raise ValueError('Fixed-context diagnostics currently require dense rewards; sparse evaluation is not silently substituted.')
    return variant


def checkpoint_paths(log_dir, epoch, names=MODEL_NAMES):
    paths = {name: Path(log_dir) / ('%s_itr_%d.pth' % (name, epoch)) for name in names}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing checkpoints (no fallback to another epoch): ' + ', '.join(missing))
    return paths


def validate_collector_variant(target, collector):
    # A cached normalized context must mean the same thing to every evaluated run.
    for key in ('env_name', 'env_params', 'seed', 'n_train_tasks', 'n_eval_tasks', 'latent_size', 'net_size'):
        if target[key] != collector[key]:
            raise ValueError('Collector run has incompatible ' + key)
    for key in ('train_epoch', 'n_trj', 'use_next_obs_in_context', 'sparse_rewards', 'max_path_length'):
        if target['algo_params'][key] != collector['algo_params'][key]:
            raise ValueError('Collector run has incompatible preprocessing: ' + key)
    if Path(target['algo_params']['data_dir']).resolve() != Path(collector['algo_params']['data_dir']).resolve():
        raise ValueError('Collector and evaluated run must use the same offline data directory/normalization')


def build_models(variant, obs_dim, action_dim, paths, device):
    import torch
    from rlkit.torch.autoencoder import MlpEncoder, MlpDecoder
    from rlkit.torch.networks import FlattenMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    latent, width = variant['latent_size'], variant['net_size']
    params = variant['algo_params']
    next_obs = params['use_next_obs_in_context']
    input_dim = obs_dim * (2 if next_obs else 1) + action_dim + 1
    factories = {
        'context_encoder': lambda: MlpEncoder(hidden_sizes=[width]*3, input_size=input_dim,
            output_size=latent*(2 if params['use_information_bottleneck'] else 1),
            output_activation=torch.tanh, batch_attention=False),
        'context_decoder': lambda: MlpDecoder(hidden_size=width, num_hidden_layers=3,
            z_dim=latent, action_dim=action_dim, obs_dim=obs_dim, reward_dim=1,
            use_next_obs_in_context=next_obs),
        'policy': lambda: TanhGaussianPolicy(hidden_sizes=[width]*3, obs_dim=obs_dim+latent,
            latent_dim=latent, action_dim=action_dim),
        'qf1': lambda: FlattenMlp(hidden_sizes=[width]*3, input_size=obs_dim+action_dim+latent, output_size=1),
        'qf2': lambda: FlattenMlp(hidden_sizes=[width]*3, input_size=obs_dim+action_dim+latent, output_size=1),
    }
    models = {}
    for name, path in paths.items():
        model = factories[name]().to(device)
        model.load_state_dict(torch.load(str(path), map_location=device, weights_only=True), strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[name] = model
    return models


def normalized_fields(fields, stats):
    import numpy as np
    result = {key: np.asarray(value).copy() for key, value in fields.items() if key in FIELDS}
    for key in ('observations', 'next_observations'):
        result[key] = (result[key] - stats['mean']) / np.sqrt(stats['var'] + 1e-8)
    return result


def resolve_observation_stats(log_dir, pools, train_tasks, eval_tasks):
    """Prefer recorded training statistics; reconstruct legacy runs explicitly."""
    import numpy as np
    from rlkit.data_management.fixed_context_data import compute_observation_stats
    reconstructed = compute_observation_stats(pools)
    path = Path(log_dir) / 'observation_normalizer.npz'
    if not path.is_file():
        print('Legacy checkpoint: reconstructing normalization from all selected training observations.', flush=True)
        return reconstructed, dict(source='reconstructed_from_current_training_data',
            note='Legacy checkpoints cannot prove that the original training dataset has not changed.')
    with np.load(path, allow_pickle=False) as stored:
        if int(stored['format_version']) != 1:
            raise ValueError('Unsupported observation normalizer format')
        for key, expected in (('train_tasks', train_tasks), ('eval_tasks', eval_tasks)):
            if not np.array_equal(stored[key], expected):
                raise ValueError('Saved observation normalizer has different ' + key)
        stats = {key: stored[key].copy() for key in ('mean', 'var', 'count')}
    for key in ('mean', 'var', 'count'):
        if np.shape(stats[key]) != np.shape(reconstructed[key]) or not np.isfinite(stats[key]).all():
            raise ValueError('Invalid saved observation normalizer ' + key)
        # File ordering changes floating-point reductions slightly, especially float32.
        if not np.allclose(stats[key], reconstructed[key], rtol=1e-5, atol=1e-6):
            raise ValueError('Training data no longer matches saved observation normalizer: ' + key)
    if (stats['var'] < 0).any():
        raise ValueError('Saved observation variance cannot be negative')
    return stats, dict(source='saved_training_normalizer', path=str(path), sha256=sha256(path))


def context_tensor(fields, next_obs, device):
    import numpy as np
    import torch
    names = ('observations', 'actions', 'rewards') + (('next_observations',) if next_obs else ())
    context = np.concatenate([fields[key] for key in names], axis=-1)
    return torch.as_tensor(context, dtype=torch.float32, device=device).unsqueeze(0)


def json_data(value):
    if isinstance(value, dict):
        return {str(k): json_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_data(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def write_json(path, value):
    # Reject NaN/Infinity instead of silently producing an invalid result artifact.
    Path(path).write_text(json.dumps(json_data(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def bank_fields(arrays, prefix):
    return {key: arrays[prefix + '__' + key] for key in FIELDS}


def save_bank(bank_dir, metadata, entries, arrays):
    import numpy as np
    bank_dir.mkdir(parents=True, exist_ok=True)
    archive = bank_dir / 'contexts.npz'
    temporary = bank_dir / 'contexts.npz.tmp'
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(archive)
    manifest = dict(metadata=metadata, entries=entries, archive_sha256=sha256(archive))
    write_json(bank_dir / 'manifest.json', manifest)
    return manifest


def load_bank(bank_dir, metadata):
    import numpy as np
    manifest = json.loads((bank_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['metadata'] != json_data(metadata):
        raise ValueError('Context bank metadata differs (data, normalizer, collectors or protocol). Use a different --bank-dir.')
    archive = bank_dir / 'contexts.npz'
    if sha256(archive) != manifest['archive_sha256']:
        raise ValueError('Context bank checksum mismatch')
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    validate_bank(metadata, manifest['entries'], arrays)
    return manifest, arrays


def validate_bank(metadata, entries, arrays):
    import numpy as np
    sources = ['offline'] + ['collector_' + str(epoch) for epoch in metadata['collector_checkpoints']]
    expected = {(source, task, repeat) for source in sources for task in metadata['tasks']
                for repeat in range(metadata['context_repeats'])}
    actual = [(entry['source'], entry['task_id'], entry['repeat']) for entry in entries]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Context bank does not contain the complete source/task/repeat product')
    if len({entry['key'] for entry in entries}) != len(entries):
        raise ValueError('Context bank contains duplicate entry keys')
    dimensions = dict(observations=len(metadata['observation_stats']['mean']),
        next_observations=len(metadata['observation_stats']['mean']), actions=metadata['action_dim'],
        rewards=1, terminals=1)
    def check(prefix, size):
        for field, width in dimensions.items():
            array = arrays.get(prefix + '__' + field)
            if array is None or array.shape != (size, width) or not np.isfinite(array).all():
                raise ValueError('Invalid paired context bank array: ' + prefix + '__' + field)
    for entry in entries:
        expected_key = '%s_task_%d_repeat_%d' % (entry['source'], entry['task_id'], entry['repeat'])
        if entry['key'] != expected_key:
            raise ValueError('Context bank key does not match its declared source/task/repeat')
        check(entry['key'], metadata['context_size'])
    for task in metadata['tasks']:
        check('probe_task_%d' % task, metadata['probe_size'])
        ids = [arrays['probe_task_%d__row_ids' % task]] + [
            arrays['offline_task_%d_repeat_%d__row_ids' % (task, repeat)]
            for repeat in range(metadata['context_repeats'])]
        flat = np.concatenate(ids)
        if len(np.unique(flat)) != len(flat):
            raise ValueError('Offline contexts and probes must have distinct row ids')


def latent_summaries(records, first_epoch):
    import numpy as np
    reference = {(r['source'], r['task_id'], r['repeat']): np.asarray(r['evaluation']['z']).reshape(-1)
                 for r in records if r['epoch'] == first_epoch}
    groups = {}
    for r in records:
        z = np.asarray(r['evaluation']['z']).reshape(-1)
        r['z_l2_from_first_checkpoint'] = float(np.linalg.norm(z - reference[(r['source'], r['task_id'], r['repeat'])]))
        groups.setdefault((r['epoch'], r['source'], r['task_id']), []).append(z)
    result = []
    for (epoch, source, task), zs in groups.items():
        z = np.stack(zs)
        mean = z.mean(0)
        result.append(dict(epoch=epoch, source=source, task_id=task, context_repeats=len(zs),
            z_mean=mean.tolist(), within_task_z_rms=float(np.sqrt(np.mean(np.sum((z-mean)**2, axis=1))))))
    return result


def evaluate_bank(env, variant, checkpoints, arrays, entries, eval_seeds, device, save_trajectories=False):
    import numpy as np
    import torch
    from rlkit.torch.fixed_context_eval import evaluate_fixed_context, diagnose_fixed_batch
    params = variant['algo_params']
    records, saved = [], {}
    for epoch, models in checkpoints.items():
        for entry in entries:
            prefix = entry['key']
            fields = bank_fields(arrays, prefix)
            context = context_tensor(fields, params['use_next_obs_in_context'], device)
            # Seeds depend only on task/replicate, never checkpoint or context source.
            reset_seeds = [(seed + 1000003*entry['task_id'] + 1009*entry['repeat']) % (2**32)
                           for seed in eval_seeds]
            evaluation, trajectories = evaluate_fixed_context(
                env, models['policy'], models['context_encoder'], context, entry['task_id'], reset_seeds,
                params['max_path_length'], params['discount'], params['reward_scale'], variant['latent_size'],
                use_information_bottleneck=params['use_information_bottleneck'], qf1=models['qf1'],
                qf2=models['qf2'], decoder=models['context_decoder'],
                use_next_obs_in_context=params['use_next_obs_in_context'])
            probes = bank_fields(arrays, 'probe_task_%d' % entry['task_id'])
            tensors = {key: torch.as_tensor(value, dtype=torch.float32, device=device)
                       for key, value in probes.items()}
            z = torch.as_tensor(evaluation['z'], dtype=torch.float32, device=device).reshape(1, -1)
            probe_summary, predictions = diagnose_fixed_batch(
                models['context_decoder'], models['qf1'], models['qf2'], tensors['observations'],
                tensors['actions'], tensors['rewards'], tensors['next_observations'], z,
                use_next_obs_in_context=params['use_next_obs_in_context'])
            records.append(dict(epoch=epoch, source=entry['source'], task_id=entry['task_id'],
                repeat=entry['repeat'], reset_seeds=reset_seeds, evaluation=evaluation, fixed_probe=probe_summary))
            if save_trajectories:
                for index, trajectory in enumerate(trajectories):
                    for key, value in trajectory.items():
                        saved['epoch_%d__%s__rollout_%d__%s' % (epoch, prefix, index, key)] = np.asarray(value)
                for key, value in predictions.items():
                    saved['epoch_%d__%s__probe__%s' % (epoch, prefix, key)] = np.asarray(value)
        print('Evaluated checkpoint %d on %d fixed contexts' % (epoch, len(entries)), flush=True)
    return records, saved


def main(argv=None):
    args = parse_args(argv)
    log_dir = args.log_dir.expanduser().resolve()
    collector_dir = (args.collector_log_dir or log_dir).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    bank_dir = (args.bank_dir or output_dir / 'context_bank').expanduser().resolve()
    for destination in (output_dir, bank_dir):
        if any(destination == run or run in destination.parents for run in (log_dir, collector_dir)):
            raise ValueError('Write diagnostic outputs outside the training/collector run directories')
    variant = load_variant(log_dir)
    if args.data_dir:
        variant['algo_params']['data_dir'] = str(args.data_dir.expanduser().resolve())
    params = variant['algo_params']
    paths = {epoch: checkpoint_paths(log_dir, epoch) for epoch in args.epochs}
    collector_paths = {} if args.offline_only else {
        epoch: checkpoint_paths(collector_dir, epoch, ('policy',)) for epoch in args.collector_epochs}
    if collector_paths:
        collector_variant = load_variant(collector_dir)
        if args.data_dir:
            collector_variant['algo_params']['data_dir'] = params['data_dir']
        validate_collector_variant(variant, collector_variant)

    # Heavy runtime imports belong here, so --help and configuration tests need no MuJoCo.
    import numpy as np
    import torch
    from rlkit.envs import ENVS
    from rlkit.envs.wrappers import NormalizedBoxEnv
    from rlkit.torch import pytorch_util as ptu
    from rlkit.torch.fixed_context_eval import collect_zero_context, isolated_evaluation
    from rlkit.data_management.fixed_context_data import load_offline_pools, build_offline_bank
    if args.gpu is not None and (not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count()):
        raise ValueError('Requested GPU is unavailable; omit --gpu for CPU evaluation')
    ptu.set_gpu_mode(args.gpu is not None, args.gpu or 0)
    env_class = ENVS[variant['env_name']]
    if 'non_mujoco' in env_class.__module__:
        raise RuntimeError('This environment is a placeholder. Configure the project MuJoCo runtime before evaluation.')
    env = NormalizedBoxEnv(env_class(**variant['env_params']))
    try:
        task_ids = list(env.get_all_task_idx())
        train_tasks = task_ids[:variant['n_train_tasks']]
        eval_tasks = task_ids[-variant['n_eval_tasks']:] if variant['n_eval_tasks'] > 0 else []
        tasks = eval_tasks if args.split == 'test' else train_tasks
        if not tasks or len(train_tasks) != variant['n_train_tasks'] or len(tasks) != variant['n_' + ('eval' if args.split == 'test' else 'train') + '_tasks']:
            raise ValueError('Requested task split is inconsistent with the environment')
        obs_dim, action_dim = int(np.prod(env.observation_space.shape)), int(np.prod(env.action_space.shape))
        train_pools, train_manifest = load_offline_pools(params['data_dir'], train_tasks, params['n_trj'],
                                                       params['train_epoch'], obs_dim, action_dim)
        stats, normalizer_origin = resolve_observation_stats(log_dir, train_pools, train_tasks, eval_tasks)
        if collector_paths and collector_dir != log_dir:
            collector_stats, _ = resolve_observation_stats(collector_dir, train_pools, train_tasks, eval_tasks)
            if any(not np.array_equal(stats[key], collector_stats[key]) for key in ('mean', 'var')):
                raise ValueError('Shared collector uses a different observation transform. Use collectors from this run; normalized banks cannot cross coordinate systems.')
        env.update_obs_mean_var(stats['mean'], stats['var'])
        if args.split == 'train':
            pools, data_manifest = train_pools, train_manifest
        else:
            pools, data_manifest = load_offline_pools(params['data_dir'], tasks, params['n_trj'],
                                                     params['eval_epoch'], obs_dim, action_dim)
        metadata = json_data(dict(protocol_version=1, env_name=variant['env_name'], env_params=variant['env_params'],
            training_seed=variant['seed'], split=args.split, tasks=tasks, train_tasks=train_tasks,
            context_size=args.context_size, context_repeats=args.context_repeats, probe_size=args.probe_size,
            bank_seed=args.bank_seed, max_path_length=params['max_path_length'], replay_capacity=params['replay_buffer_size'],
            observation_stats=stats, training_data=train_manifest, context_data=data_manifest,
            latent_size=variant['latent_size'], action_dim=action_dim,
            use_next_obs_in_context=params['use_next_obs_in_context'],
            collector_checkpoints={str(epoch): dict(path=str(p['policy']), sha256=sha256(p['policy']))
                                   for epoch, p in collector_paths.items()}))
        if (bank_dir / 'manifest.json').exists() or (bank_dir / 'contexts.npz').exists():
            manifest, arrays = load_bank(bank_dir, metadata)
            entries = manifest['entries']
            print('Reusing verified context bank: %s' % bank_dir, flush=True)
        else:
            bank = build_offline_bank(pools, args.context_size, args.context_repeats, args.probe_size,
                                      args.bank_seed, capacity=params['replay_buffer_size'])
            arrays, entries = {}, []
            for task, item in bank.items():
                for key, value in normalized_fields(item['probes'], stats).items():
                    arrays['probe_task_%d__%s' % (task, key)] = value
                arrays['probe_task_%d__row_ids' % task] = item['probe_row_ids']
                for repeat in range(args.context_repeats):
                    prefix = 'offline_task_%d_repeat_%d' % (task, repeat)
                    fields = {key: item['contexts'][key][repeat] for key in FIELDS}
                    for key, value in normalized_fields(fields, stats).items():
                        arrays[prefix + '__' + key] = value
                    arrays[prefix + '__row_ids'] = item['context_row_ids'][repeat]
                    entries.append(dict(key=prefix, source='offline', task_id=task, repeat=repeat))
            for epoch, checkpoint in collector_paths.items():
                with isolated_evaluation([], seed=variant['seed']):
                    policy = build_models(variant, obs_dim, action_dim, checkpoint, ptu.device)['policy']
                for task in tasks:
                    for repeat in range(args.context_repeats):
                        prefix = 'collector_%d_task_%d_repeat_%d' % (epoch, task, repeat)
                        seed = (args.bank_seed + 1000003*task + 1009*repeat) % (2**32)
                        fields = collect_zero_context(env, policy, task, seed, args.context_size,
                                                      params['max_path_length'], variant['latent_size'])
                        for key, value in fields.items():
                            arrays[prefix + '__' + key] = value
                        entries.append(dict(key=prefix, source='collector_%d' % epoch,
                                            task_id=task, repeat=repeat, collection_seed=seed))
                print('Collected and froze online contexts from checkpoint %d' % epoch, flush=True)
            validate_bank(metadata, entries, arrays)
            manifest = save_bank(bank_dir, metadata, entries, arrays)
        # Free raw replay arrays before loading all requested networks.
        del pools, train_pools
        with isolated_evaluation([], seed=variant['seed']):
            models = {epoch: build_models(variant, obs_dim, action_dim, p, ptu.device) for epoch, p in paths.items()}
        records, trajectories = evaluate_bank(env, variant, models, arrays, entries, args.eval_seeds,
                                               ptu.device, args.save_trajectories)
        result = dict(protocol='fixed-context-v1', log_dir=str(log_dir), variant_sha256=sha256(log_dir / 'variant.json'),
            checkpoint_files={str(epoch): {name: dict(path=str(p), sha256=sha256(p)) for name, p in files.items()}
                              for epoch, files in paths.items()}, bank_dir=str(bank_dir),
            bank_archive_sha256=manifest['archive_sha256'], observation_normalizer=normalizer_origin, settings=vars(args),
            notes=['Every context is encoded once; z never changes during evaluation rollouts.',
                   'Online collectors use stochastic current policy with z=0; evaluation uses deterministic actions.',
                   'IB models use posterior means here, not historical posterior sampling.',
                   'Q errors compare scaled, finite-horizon on-policy returns; truncation omits the continuation tail.',
                   'Offline probe Q values have no Monte Carlo target; offline behavior returns are not policy Q ground truth.',
                   'Latent distance is a diagnostic of fixed-context change, not proof of semantic collapse.',
                   'Offline row ids are disjoint between context repeats and probes; source data may itself contain duplicates.'],
            latent_statistics=latent_summaries(records, args.epochs[0]), records=records)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / 'fixed_context_results.json', result)
        import csv
        with (output_dir / 'returns.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['epoch', 'source', 'task_id', 'context_repeat', 'return_mean', 'reset_seed_std',
                             'z_l2_from_first_checkpoint'])
            for r in records:
                writer.writerow([r['epoch'], r['source'], r['task_id'], r['repeat'], r['evaluation']['return_mean'],
                                 r['evaluation']['return_std'], r['z_l2_from_first_checkpoint']])
        if args.save_trajectories:
            np.savez_compressed(output_dir / 'trajectory_diagnostics.npz', **trajectories)
        print('Saved fixed-context results: %s' % output_dir, flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
