"""Collect one task's mixed Random/Medium/Expert offline trajectories."""

import argparse
import importlib
from pathlib import Path
import re
import sys
import tempfile


ENVIRONMENTS = (
    'point-robot', 'ant-dir', 'cheetah-vel', 'cheetah-dir',
    'hopper-rand-params', 'walker-rand-params',
)
PROTOCOL = 'shared-native-v3'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=ENVIRONMENTS, required=True)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--focal-root', type=Path, required=True)
    parser.add_argument('--task-manifest', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True,
                        help='Root containing <env>/goal_idx<id>/actor_step<step>.pkl')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New mixed trajectory root, separate from historical data')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--calibration-episodes', type=int, default=20)
    args = parser.parse_args()
    if args.calibration_episodes <= 0:
        parser.error('--calibration-episodes must be positive')
    return args


def collect(args):
    import numpy as np
    import torch
    import yaml

    sys.path.insert(1, str(args.focal_root.resolve()))
    from focal_v2.common import benchmark, evaluation_seed, file_hash, write_json
    from focal_v2.environments import ALIAS, export_or_check_tasks, make_env
    from rlkit.torch.sac.pytorch_sac.agent.actor import DiagGaussianActor

    cfg = benchmark(args.env)
    if args.task not in cfg['train_tasks']:
        raise ValueError(f'{args.env}: task {args.task} is outside the training split')

    suffix = Path(args.env) / f'goal_idx{args.task}'
    checkpoint_dir = (args.checkpoint_root / suffix).resolve()
    destination = (args.output_root / suffix).resolve()
    if checkpoint_dir == destination or destination.exists():
        raise FileExistsError(f'Choose a fresh mixed trajectory directory: {destination}')

    checkpoints = {}
    for path in checkpoint_dir.glob('actor_step*.pkl'):
        match = re.fullmatch(r'actor_step([1-9]\d*)\.pkl', path.name)
        if match and int(match[1]) <= cfg['data_checkpoint']:
            checkpoints[int(match[1])] = path
    expert_step = cfg['data_checkpoint']
    if expert_step not in checkpoints or len(checkpoints) < 2:
        raise FileNotFoundError(f'{checkpoint_dir}: need actor_step{expert_step}.pkl and intermediate actors')

    task_manifest_hash = file_hash(args.task_manifest)
    raw_env = make_env(cfg, args.reference_root)
    try:
        export_or_check_tasks(raw_env, cfg, args.task_manifest)
        raw_env.reset_task(args.task)
        wrappers = importlib.import_module(ALIAS + '.wrappers')
        env = wrappers.NormalizedBoxEnv(raw_env)
        actor_config = (Path(__file__).resolve().parent /
                        'rlkit/torch/sac/pytorch_sac/config/agent/sac.yaml')
        params = yaml.safe_load(actor_config.read_text(encoding='utf-8'))['diag_gaussian_actor']['params']
        actor = DiagGaussianActor(
            obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
            hidden_dim=params['hidden_dim'], hidden_depth=params['hidden_depth'],
            log_std_bounds=params['log_std_bounds'],
        ).eval()

        def seed_for(stage, index):
            return evaluation_seed(PROTOCOL, args.env, args.seed, args.task, stage, index)

        def rollout(seed, policy):
            env.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            rng = np.random.RandomState(seed)
            obs = env.reset()
            rows = []
            done = False
            with torch.no_grad():
                for _ in range(cfg['horizon']):
                    if policy is None:
                        action = rng.uniform(env.action_space.low, env.action_space.high)
                    else:
                        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                        action = policy(obs_tensor).sample()[0].cpu().numpy()
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    saved_obs, saved_action = np.array(obs, copy=True), action.copy()
                    obs, reward, done, _ = env.step(action)
                    rows.append((saved_obs, saved_action, float(reward), np.array(obs, copy=True)))
                    if done:
                        break
            if any(not np.isfinite(field).all() for row in rows for field in row):
                raise ValueError(f'{suffix}: non-finite rollout with seed {seed}')
            return rows, bool(done), sum(row[2] for row in rows)

        def load_actor(step):
            actor.load_state_dict(torch.load(checkpoints[step], map_location='cpu', weights_only=True))
            return actor

        calibration_seeds = [seed_for('calibration', i) for i in range(args.calibration_episodes)]

        def calibrate(policy):
            return [rollout(seed, policy)[2] for seed in calibration_seeds]

        calibration = {None: calibrate(None)}
        checkpoint_hashes = {step: file_hash(path) for step, path in checkpoints.items()}
        for step in sorted(checkpoints):
            calibration[step] = calibrate(load_actor(step))
        means = {step: float(np.mean(values)) for step, values in calibration.items()}
        improvement = means[expert_step] - means[None]
        if improvement <= 0:
            raise ValueError(f'{suffix}: Expert calibration return did not exceed Random')
        fractions = {step: (means[step] - means[None]) / improvement
                     for step in checkpoints if step < expert_step}
        medium_step = min(fractions, key=lambda step: (abs(fractions[step] - 0.5), step))
        if not 0.35 <= fractions[medium_step] <= 0.65:
            raise ValueError(f'{suffix}: no Medium checkpoint in [0.35, 0.65]; candidates {fractions}')

        quotas = (34, 33, 33) if cfg['n_trj'] == 100 else (17, 17, 16)
        slots = np.random.RandomState(seed_for('shuffle', 0)).permutation(cfg['n_trj'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.rme-', dir=destination.parent) as temporary:
            staged = Path(temporary) / 'task'
            staged.mkdir()
            records = []
            for behavior, step, quota in zip(
                    ('random', 'medium', 'expert'), (None, medium_step, expert_step), quotas):
                policy = None if step is None else load_actor(step)
                for _ in range(quota):
                    index = int(slots[len(records)])
                    seed = seed_for('collection', index)
                    if seed in calibration_seeds or any(item['seed'] == seed for item in records):
                        raise ValueError(f'{suffix}: collection seed collision')
                    rows, done, episode_return = rollout(seed, policy)
                    filename = f'trj_evalsample{index}.npy'
                    data = np.empty((len(rows), 4), dtype=object)
                    for row_index, row in enumerate(rows):
                        for column, field in enumerate(row):
                            data[row_index, column] = field
                    np.save(staged / filename, data)
                    records.append(dict(
                        path=(suffix / filename).as_posix(), trajectory_index=index,
                        task_id=args.task, behavior=behavior, checkpoint=step,
                        checkpoint_sha256=checkpoint_hashes.get(step), seed=seed,
                        rows=len(rows), sha256=file_hash(staged / filename),
                        environment_done=done, episode_return=episode_return,
                    ))
            if (file_hash(args.task_manifest) != task_manifest_hash
                    or any(file_hash(checkpoints[step]) != sha for step, sha in checkpoint_hashes.items())):
                raise ValueError(f'{suffix}: task manifest or checkpoint changed during collection')
            write_json(staged / 'collection.json', dict(
                protocol_id=PROTOCOL, environment=args.env, task_id=args.task,
                task_manifest_sha256=task_manifest_hash, backend=raw_env.shared_backend,
                action_low=raw_env.action_space.low, action_high=raw_env.action_space.high,
                data_action_coordinates='normalized', collection_seed=args.seed,
                calibration_seeds=calibration_seeds,
                calibration_returns={str(step): values for step, values in calibration.items()},
                medium_fraction=fractions[medium_step], files=sorted(
                    records, key=lambda entry: entry['trajectory_index']),
            ))
            staged.rename(destination)
    finally:
        raw_env.close()
    print(destination)


if __name__ == '__main__':
    collect(parse_args())
