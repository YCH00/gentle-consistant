"""Train early single-task SAC actors and stage the historical Expert for v3."""

import argparse
import importlib
from pathlib import Path
import shutil
import sys
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--focal-root', type=Path, required=True)
    parser.add_argument('--task-manifest', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--env', default='point-robot')
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.steps <= 0 or args.save_every <= 0:
        parser.error('--steps and --save-every must be positive')
    return args


def recover(args):
    from omegaconf import OmegaConf

    reference = args.reference_root.resolve()
    sys.path.insert(1, str(args.focal_root.resolve()))
    from focal_v2.common import benchmark, file_hash, write_json
    from focal_v2.environments import ALIAS, export_or_check_tasks, make_env

    benchmark_cfg = benchmark(args.env)
    if args.task not in benchmark_cfg['train_tasks']:
        raise ValueError(f'{args.env}: task {args.task} is outside the training split')
    expert_step = benchmark_cfg['data_checkpoint']
    suffix = Path(args.env) / f'goal_idx{args.task}'
    historical_task = (args.checkpoint_root / suffix).resolve()
    expert = historical_task / f'actor_step{expert_step}.pkl'
    destination = (args.output_root / suffix).resolve()
    if not expert.is_file():
        raise FileNotFoundError(expert)
    if destination == historical_task or destination.exists():
        raise FileExistsError(f'Choose a fresh recovery output directory: {destination}')

    config_root = reference / 'rlkit/torch/sac/pytorch_sac/config'
    cfg = OmegaConf.merge(OmegaConf.load(str(config_root / 'train.yaml')),
                          OmegaConf.load(str(config_root / 'agent/sac.yaml')))
    if not cfg.num_seed_steps < args.steps < expert_step:
        raise ValueError(f'--steps must be after warmup ({cfg.num_seed_steps}) and before Expert ({expert_step})')
    save_steps = list(range((cfg.num_seed_steps // args.save_every + 1) * args.save_every,
                            args.steps + 1, args.save_every))
    if not save_steps:
        raise ValueError('The requested save frequency leaves no post-warmup checkpoint')
    cfg.num_train_steps = args.steps
    cfg.eval_frequency = expert_step // 100
    cfg.save_frequency = args.save_every
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.log_save_tb = False
    cfg.save_video = False

    task_hash = file_hash(args.task_manifest)
    expert_hash = file_hash(expert)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.medium-', dir=destination.parent) as scratch:
        scratch = Path(scratch)
        # Workspace derives its data root from the location of its own source file.
        shutil.copytree(reference / 'rlkit', scratch / 'rlkit',
                        ignore=shutil.ignore_patterns('__pycache__'))
        sys.path.insert(0, str(scratch))
        from rlkit.torch.sac.pytorch_sac.train import Workspace

        raw_env = make_env(benchmark_cfg, reference)
        try:
            export_or_check_tasks(raw_env, benchmark_cfg, args.task_manifest)
            wrappers = importlib.import_module(ALIAS + '.wrappers')
            env = wrappers.NormalizedBoxEnv(raw_env)
            env.seed(args.seed)
            env.action_space.seed(args.seed)
            raw_env.reset_task(args.task)
            workspace = Workspace(cfg, args.env, env=env, goal_idx=args.task)
            workspace.run()
        finally:
            raw_env.close()

        staged = scratch / 'actors'
        staged.mkdir()
        recovered = {}
        for step in save_steps:
            filename = f'actor_step{step}.pkl'
            (scratch / 'data' / suffix / filename).rename(staged / filename)
            recovered[str(step)] = file_hash(staged / filename)
        shutil.copyfile(expert, staged / expert.name)
        if file_hash(staged / expert.name) != expert_hash or file_hash(args.task_manifest) != task_hash:
            raise ValueError('Historical Expert or task manifest changed during recovery')
        write_json(staged / 'medium_recovery.json', dict(
            environment=args.env, task_id=args.task, training_seed=args.seed,
            training_steps=args.steps, save_every=args.save_every, warmup_steps=cfg.num_seed_steps,
            task_manifest_sha256=task_hash, recovered_actor_sha256=recovered,
            historical_expert_sha256=expert_hash, backend=raw_env.shared_backend,
        ))
        staged.rename(destination)
    print(destination)


if __name__ == '__main__':
    recover(parse_args())
