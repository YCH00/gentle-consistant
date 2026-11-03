"""One isolated method/environment process; real simulator smoke only."""
import argparse
import glob
import json
import os
from pathlib import Path
import sys
import time
import traceback

from audit import sha256, write_json


def reconstruct_training_stats(entry, plan, out):
    import numpy as np
    from runtime_common import native_rms, load_stats
    cfg = entry['configuration']
    a = cfg['algo_params']
    shared = plan['shared'][entry['environment']]
    manifest = json.loads(Path(shared['dataset_manifest.json']['path']).read_text())
    task = shared['task_payload']
    if task['train_tasks'] != list(range(cfg['n_train_tasks'])) or task['eval_tasks'] != list(range(len(task['tasks']) - cfg['n_eval_tasks'], len(task['tasks']))):
        raise ValueError('Legacy training task split differs from the shared split')
    if manifest['data_action_coordinates'] != 'normalized':
        raise ValueError('Legacy training actions were not in the expected normalized coordinates')
    root = Path(a['data_dir'])
    verified = []
    for row in manifest['files']:
        path = (root / row['path']).resolve()
        if root.resolve() not in path.parents or path.parent.name != 'goal_idx' + str(row['task_id']):
            raise ValueError('Offline file task identity/path invalid')
        if sha256(path) != row['sha256']:
            raise ValueError('Original offline pool changed: ' + str(path))
        data = np.load(path, allow_pickle=True)
        if data.shape != (row['rows'], 4):
            raise ValueError('Original offline schema differs: ' + str(path))
        for col, key in enumerate(('observation', 'action', 'reward', 'next_observation')):
            values = np.asarray(list(data[:, col])).reshape(row['rows'], -1)
            if values.shape[1] != row['dimensions'][key] or not np.isfinite(values).all():
                raise ValueError('Invalid original offline field: ' + key)
            if key == 'action' and np.max(np.abs(values)) > 1 + 1e-6:
                raise ValueError('Offline actions are not normalized')
        verified.append(dict(path=str(path), sha256=row['sha256'], task_id=row['task_id'], rows=row['rows']))
    if {f['task_id'] for f in verified} != set(task['train_tasks'] + task['eval_tasks']):
        raise ValueError('Original offline pool does not cover all manifest tasks')
    write_json(out / 'offline_pool_verification.json', dict(files=verified,
        absorbing_terminal_status='unknown in original four-column format',
        native_episode_boundary='last row of each original trajectory; not an absorbing-terminal label'))
    source = {str((root / f['path']).resolve()): f for f in manifest['files'] if f['task_id'] in task['train_tasks']}
    paths = []
    for trajectory in range(a['n_trj']):
        pattern = 'trj_evalsample{}_step{}.npy'.format(trajectory, int(a['train_epoch']))
        paths.extend(p for p in glob.glob(str(root / 'goal_idx*' / pattern))
                     if int(Path(p).parent.name[len('goal_idx'):]) in task['train_tasks'])
    if {str(Path(p).resolve()) for p in paths} != set(source) or len(paths) != len(source):
        raise ValueError('Native training files differ from shared manifest; no normalizer reconstruction permitted')
    observations, order = [], []
    for path in paths:
        row = source[str(Path(path).resolve())]
        if sha256(path) != row['sha256']:
            raise ValueError('Original training data changed: ' + str(path))
        # The original local offline data uses NumPy object arrays. Its source
        # is explicitly identified and hash-checked above; no downloaded pickle.
        data = np.load(path, allow_pickle=True)
        if len(data) != row['rows']:
            raise ValueError('Training row count differs')
        observations.extend(list(data[:, 0]))
        order.append({'path': str(path), 'sha256': row['sha256'], 'rows': row['rows']})
    original = native_rms(np.asarray(observations))
    published = load_stats(shared['observation_normalizer.npz']['path'])
    if any(not np.isfinite(v).all() for v in original.values()):
        raise ValueError('Non-finite reconstructed training statistics')
    np.savez(out / 'legacy_training_normalizer.npz', **original)
    contract = {key: a[key] for key in ('data_dir', 'n_trj', 'train_epoch')}
    receipt = {'kind': 'reconstruction_of_original_training_update', 'test_data_used': False,
               'evaluation_data_used': False, 'native_loader_order': order, 'training_contract': contract,
               'historical_limit': 'Original run did not save RMS or filesystem enumeration order; reconstruction uses the unchanged source files and current native glob order; compare shared RMS explicitly',
               'published_stats_max_abs_difference': {k: float(np.max(np.abs(original[k] - published[k]))) for k in original},
               'published_stats_close': all(np.allclose(original[k], published[k], rtol=1e-5, atol=1e-6) for k in original),
               'normalizer_sha256': sha256(out / 'legacy_training_normalizer.npz')}
    write_json(out / 'normalizer_provenance.json', receipt)
    return original, receipt


def prepare_bank(plan, selection, environment, out, device):
    import numpy as np
    import torch
    from native_adapters import NativeAdapter
    from real_environment import make_environment, reset_task, step
    from runtime_common import isolated_rng, seed32, context_hash, weights_hash
    entry = next(e for e in selection['models'] if e['method'] == 'GENTLE' and e['environment'] == environment and e['training_seed'] == 0)
    stats, provenance = reconstruct_training_stats(entry, plan, out)
    env, backend = make_environment(plan, environment)
    try:
        adapter = NativeAdapter(entry, plan['shared'][environment], env.observation_space.shape[0], env.action_space.shape[0], device, stats)
        frozen = weights_hash(adapter.modules)
        task = sorted(plan['shared'][environment]['task_payload']['train_tasks'])[0]
        seeds, contexts, index = [], {}, []
        for repeat in range(2):
            seed = seed32(plan['protocol']['protocol_id'], environment, task, 'smoke_collection_reset', repeat=repeat)
            values = {key: [] for key in ('observations', 'actions', 'rewards', 'next_observations', 'episode_end')}
            resets, length = [{'at_row': 0, 'seed': seed}], 0
            with isolated_rng(seed):
                obs = reset_task(env, task, seed)
                prior = torch.zeros((1, adapter.latent_dim), device=adapter.device)
                for row in range(200):
                    action = adapter.act(obs, prior)
                    nxt, reward, done, _ = step(env, action)
                    length += 1
                    boundary = done or length == 200
                    for key, value in zip(values, (obs.copy(), action.copy(), [reward], nxt.copy(), [int(boundary)])):
                        values[key].append(value)
                    obs = nxt
                    if boundary and row < 199:
                        continuation = seed32(plan['protocol']['protocol_id'], environment, task, 'smoke_collection_reset', repeat=repeat, rollout=len(resets))
                        obs = reset_task(env, task, continuation)
                        resets.append({'at_row': row + 1, 'seed': continuation})
                        length = 0
            raw = {k: np.asarray(v) for k, v in values.items()}
            for key, value in raw.items():
                contexts['c{}_{}'.format(repeat, key)] = value
            index.append({'context_id': 'smoke-{}'.format(repeat), 'context_repeat': repeat, 'task_id': task,
                          'task_split': 'seen', 'condition_id': 'policy499_sigma000', 'context_hash': context_hash(raw),
                          'context_length': 200, 'collection_seed': seed, 'resets': resets})
        if weights_hash(adapter.modules) != frozen:
            raise ValueError('Collector weights changed')
        np.savez_compressed(out / 'contexts.npz', **contexts)
        write_json(out / 'manifest.json', {'protocol': plan['protocol']['protocol_id'], 'kind': 'smoke_only',
                   'environment': environment, 'observation_space': 'raw', 'action_space': 'normalized', 'reward_space': 'raw',
                   'terminal_semantics': 'episode_end is the native boundary channel, not absorbing-terminal ground truth',
                   'collector': entry, 'collector_latent': 'zero', 'collector_actor': 'deterministic',
                   'backend': backend, 'contexts_sha256': sha256(out / 'contexts.npz'), 'contexts': index,
                   'normalizer_provenance': provenance})
    finally:
        env.close()


def smoke(plan, selection, environment, method, seed, bank, out, device, collector200=False):
    import numpy as np
    import torch
    from native_adapters import NativeAdapter
    from real_environment import make_environment, reset_task, step
    from runtime_common import isolated_rng, seed32, weights_hash, context_hash, load_stats, digest
    entry = next(e for e in selection['collectors' if collector200 else 'models'] if e['environment'] == environment and e['method'] == method and e['training_seed'] == seed)
    manifest = json.loads((bank / 'manifest.json').read_text())
    if sha256(bank / 'contexts.npz') != manifest['contexts_sha256']:
        raise ValueError('Shared smoke bank changed')
    stats = None
    normalizer = None
    if method in plan['protocol']['run_selection']:
        provenance = json.loads((bank / 'normalizer_provenance.json').read_text())
        if {k: entry['configuration']['algo_params'][k] for k in provenance['training_contract']} != provenance['training_contract']:
            raise ValueError('Legacy model used a different training data contract')
        normalizer = bank / 'legacy_training_normalizer.npz'
        if sha256(normalizer) != provenance['normalizer_sha256']:
            raise ValueError('Native training statistics changed')
        stats = load_stats(normalizer)
    elif method == 'CSRO':
        normalizer = Path(plan['shared'][environment]['observation_normalizer.npz']['path'])
        stats = load_stats(normalizer)
    env, backend = make_environment(plan, environment)
    try:
        adapter = NativeAdapter(entry, plan['shared'][environment], env.observation_space.shape[0], env.action_space.shape[0], device, stats)
        before, stats_before = weights_hash(adapter.modules), digest(stats) if stats else None
        episodes, embeddings, indices = [], [], []
        started = time.perf_counter()
        with np.load(bank / 'contexts.npz', allow_pickle=False) as data:
            for item in manifest['contexts']:
                repeat, task = item['context_repeat'], item['task_id']
                raw = {key: data['c{}_{}'.format(repeat, key)].copy() for key in ('observations', 'actions', 'rewards', 'next_observations', 'episode_end')}
                if context_hash(raw) != item['context_hash']:
                    raise ValueError('Raw context array hash mismatch')
                posterior_seed = seed32(plan['protocol']['protocol_id'], environment, task,
                                        'posterior:{}:{}'.format(method, seed), 'policy499_sigma000', repeat)
                with isolated_rng(posterior_seed):
                    representation = adapter.infer(raw, representation=True).cpu().numpy()
                with isolated_rng((posterior_seed + 1) % 2**32):
                    again = adapter.infer(raw, representation=True).cpu().numpy()
                if not np.array_equal(representation, again) or not np.isfinite(representation).all():
                    raise ValueError('Deterministic representation is unstable/non-finite')
                with isolated_rng(posterior_seed):
                    condition = adapter.infer(raw, representation=False)
                if collector200:
                    condition = torch.zeros_like(condition)
                reset_seed = seed32(plan['protocol']['protocol_id'], environment, task, 'evaluation_reset', repeat=repeat)
                rewards, actions = [], []
                with isolated_rng(reset_seed):
                    observation = reset_task(env, task, reset_seed)
                    for length in range(1, 201):
                        action = adapter.act(observation, condition)
                        observation, reward, done, _ = step(env, action)
                        rewards.append(reward)
                        actions.append(action.tolist())
                        if done:
                            break
                episodes.append(dict(protocol=plan['protocol']['protocol_id'], method=method, environment=environment,
                    training_seed=seed, checkpoint_hashes=[f['sha256'] for f in entry['checkpoint_files']],
                    **item, rollout_id=0, reset_seed=reset_seed, posterior_seed=posterior_seed,
                    episode_return_raw=float(sum(rewards)), episode_length=length,
                    done_reason='environment_done' if done else 'horizon', raw_rewards=rewards, normalized_actions=actions,
                    native_policy_condition=condition.cpu().numpy().tolist(), task_manifest_hash=plan['shared'][environment]['task_manifest.json']['sha256']))
                episodes[-1].update(normalizer_file_sha256=sha256(normalizer) if normalizer else None,
                    preprocessing='identity' if stats is None else 'native_RMS_once',
                    collector_checkpoint_hashes=[f['sha256'] for f in manifest['collector']['checkpoint_files']],
                    source_evidence_sha256=plan['evidence_sha256'], model_role=entry.get('role', 'evaluation'))
                embeddings.append(representation[0])
                indices.append(dict(item, method=method, training_seed=seed, latent_kind=adapter.report['latent_kind']))
        if weights_hash(adapter.modules) != before or (digest(stats) if stats else None) != stats_before:
            raise ValueError('Model weights or statistics mutated during smoke')
        np.savez_compressed(out / 'embeddings.npz', embeddings=np.stack(embeddings))
        write_json(out / 'embedding_index.json', indices)
        with (out / 'episodes.jsonl').open('w', encoding='utf-8') as stream:
            for episode in episodes:
                stream.write(json.dumps(episode, allow_nan=False) + '\n')
        write_json(out / 'completion.json', {'smoke_passed': True, 'formal_result': False, 'episodes': len(episodes),
                   'stage_A_complete': False, 'entry': entry, 'adapter': adapter.report, 'backend': backend,
                   'raw_context_hashes': [item['context_hash'] for item in manifest['contexts']],
                   'weights_before_and_after_sha256': before, 'normalizer_statistics_sha256': stats_before,
                   'elapsed_seconds': time.perf_counter() - started,
                   'output_sha256': {p.name: sha256(p) for p in out.iterdir() if p.is_file()}})
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--environment', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--collector200', action='store_true')
    parser.add_argument('--method')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--mujoco131')
    parser.add_argument('--mujoco150')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        plan, selection = json.loads(args.plan.read_text()), json.loads(args.selection.read_text())
        from real_environment import configure_backend
        configure_backend(args.environment, args.mujoco131 if args.environment == 'walker-rand-params' else args.mujoco150)
        import torch
        torch.set_num_threads(1)
        if args.prepare:
            prepare_bank(plan, selection, args.environment, args.out, args.device)
        else:
            smoke(plan, selection, args.environment, args.method, args.seed, args.bank, args.out, args.device, args.collector200)
        print('PASS', args.environment, 'shared_bank' if args.prepare else args.method, args.seed, flush=True)
        return 0
    except Exception as exc:
        write_json(args.out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
                                              'smoke_passed': False, 'stage_A_complete': False})
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
