"""Atomic, independently verifiable exports. This module never trains a model."""
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from shared_protocol.common import PROTOCOL, digest, file_hash, read_json, write_json
from shared_protocol.data import load_raw, fit_normalizer, load_normalizer, transform_pools
from shared_protocol.environments import make_env, export_or_check_tasks, task_payload
from shared_protocol.runtime import ROOT, CORE_FILES, ENV_NAMES, load_bundle, protocol_config


def backend_smoke(env, cfg):
    state = np.random.get_state()
    try:
        np.random.seed(1337)
        for task in sorted(set(cfg['train_tasks'] + cfg['eval_tasks'])):
            env.reset_task(task)
            obs = env.reset()
            next_obs, reward, done, info = env.step(np.zeros(env.action_space.shape))
            if (np.asarray(obs).shape != env.observation_space.shape or
                    np.asarray(next_obs).shape != env.observation_space.shape or
                    not np.isfinite(obs).all() or not np.isfinite(next_obs).all() or not np.isfinite(reward)):
                raise ValueError('Invalid real backend output at task {}'.format(task))
        env.reset_task(0)
    finally:
        np.random.set_state(state)
    return dict(status='passed', tasks_checked=len(set(cfg['train_tasks'] + cfg['eval_tasks'])),
                check='real reset and one physical zero-action step per task')


def history_evidence(env, cfg):
    tasks = task_payload(env, cfg)['tasks']
    records = []
    for task in range(len(tasks)):
        path = Path(cfg['data_dir']) / ('goal_idx{}'.format(task)) / 'log.npy'
        if not path.exists():
            records.append(dict(task_id=task, status='missing'))
            continue
        saved = np.load(path, allow_pickle=True)
        if saved.shape == ():
            saved = saved.item()
        if isinstance(saved, dict) and 'task' in saved:
            saved = saved['task']
        if isinstance(saved, dict) and 'goal' in saved and cfg['environment'] == 'point-robot':
            saved = saved['goal']
        expected = tasks[task]
        recognizable = (isinstance(expected, dict) and isinstance(saved, dict) and
                        set(expected).issubset(saved))
        if recognizable:
            saved = {key: saved[key] for key in expected}
        elif cfg['environment'] == 'point-robot':
            recognizable = np.asarray(saved).shape == (2,) and np.asarray(saved).dtype.kind in 'fiu'
        if recognizable:
            if digest(saved) != digest(expected):
                raise ValueError('Historical task record conflicts with reference task {}: {}'.format(task, path))
            status = 'matched'
        else:
            status = 'unrecognized_format_requires_manual_review'
        records.append(dict(task_id=task, path=str(path), sha256=file_hash(path), status=status))
    return dict(status='verified' if all(r['status'] == 'matched' for r in records) else 'not_fully_verified',
                basis='reconstructed from pinned reference code; historical records checked only where recognizable',
                records=records)


def source_provenance():
    paths = sorted(set(list((ROOT / 'shared_protocol').glob('*.*')) +
                       [ROOT / 'export_shared_artifacts.py', ROOT / 'train_gentle.py',
                        ROOT / 'pretrain_dynamics.py', ROOT / 'pretrain_encoder_decoder.py',
                        ROOT / 'rlkit/core/rl_algorithm.py', ROOT / 'evaluate_fixed_context.py',
                        ROOT / 'policy_eval.py', ROOT / 'rlkit/torch/sac/pytorch_sac/train.py',
                        ROOT / 'rlkit/envs/wrappers.py',
                        ROOT / 'rlkit/data_management/fixed_context_data.py']))
    sources = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths if p.is_file()}
    versions = {'python': sys.version}
    for name in ('numpy', 'gym', 'torch', 'mujoco-py'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    def git(*args):
        result = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        commit, status = git('rev-parse', 'HEAD'), git('status', '--porcelain')
    except FileNotFoundError:
        commit, status = None, 'git unavailable'
    return dict(actual_commit=commit, worktree_status=status, tool_sources=sources,
                tool_sources_digest=digest(sources), versions=versions)


def export_environment(name, data_dir, output_root, action_coordinates, action_evidence,
                       normalizer=None, task_manifest=None):
    cfg = protocol_config(name, data_dir)
    root = Path(output_root).expanduser().resolve()
    target = root / name
    data_path = Path(cfg['data_dir'])
    if target == data_path or data_path in target.parents:
        raise ValueError('Shared outputs must not be inside the raw data directory')
    if not data_path.is_dir():
        raise FileNotFoundError('Raw data directory does not exist: ' + str(data_path))
    env = make_env(cfg, ROOT)
    temporary = None
    lock = root / ('.' + name + '.export-lock')
    locked = False
    try:
        backend = backend_smoke(env, cfg)
        history = history_evidence(env, cfg)
        if target.exists():
            # Reuse only a complete, valid bundle. Do not replace any published bytes.
            for filename in (*CORE_FILES, 'validation.json', 'preparation_provenance.json'):
                if not (target / filename).is_file():
                    raise FileExistsError('Incomplete existing export; choose a new output root: ' + str(target))
            bundle = load_bundle(cfg, env, target)
            if bundle['manifest']['data_action_coordinates'] != action_coordinates:
                raise ValueError('Existing export has different action coordinates')
            for candidate, filename in ((normalizer, 'observation_normalizer.npz'), (task_manifest, 'task_manifest.json')):
                if candidate and file_hash(candidate) != bundle['hashes'][filename]:
                    raise ValueError('Requested shared input differs from existing export: ' + filename)
            validation = read_json(target / 'validation.json')
            if validation.get('artifact_status') != 'passed':
                raise ValueError('Existing export was not marked validated')
            return target, 'verified_and_reused'
        root.mkdir(parents=True, exist_ok=True)
        lock.mkdir()  # fail if another process is publishing this environment
        locked = True
        temporary = Path(tempfile.mkdtemp(prefix='.' + name + '-staging-', dir=root))
        if task_manifest:
            export_or_check_tasks(env, cfg, task_manifest)
            shutil.copyfile(task_manifest, temporary / 'task_manifest.json')
        else:
            export_or_check_tasks(env, cfg, temporary / 'task_manifest.json', create=True)
        pools, manifest = load_raw(cfg, env.observation_space.shape[0], env.action_space.shape[0], action_coordinates)
        expected = fit_normalizer(pools, cfg['train_tasks'])
        stats = load_normalizer(normalizer, expected) if normalizer else expected
        normalizer_path = temporary / 'observation_normalizer.npz'
        if normalizer:
            shutil.copyfile(normalizer, normalizer_path)
        else:
            np.savez_compressed(normalizer_path, **stats, format_version=np.asarray(1),
                                train_tasks=np.asarray(cfg['train_tasks']), eval_tasks=np.asarray(cfg['eval_tasks']))
        # Explicit coordinate/range and finite float32 checks, without changing raw files.
        transform_pools(pools, stats, action_coordinates, env.action_space.low, env.action_space.high)
        manifest['normalizer_sha256'] = file_hash(normalizer_path)
        manifest['task_manifest_sha256'] = file_hash(temporary / 'task_manifest.json')
        write_json(temporary / 'dataset_manifest.json', manifest)
        dtypes = {str(t): {k: str(v.dtype) for k, v in p.items()} for t, p in pools.items()}
        del pools
        bundle = load_bundle(cfg, env, temporary)
        write_json(temporary / 'preparation_provenance.json', dict(
            exported_at_utc=datetime.now(timezone.utc).isoformat(), protocol_id=PROTOCOL, environment=name,
            command=sys.argv, source=source_provenance(), backend=env.shared_backend,
            data_dir=cfg['data_dir'], reading_order='numeric task ID, numeric trajectory index, original row',
            field_dtypes=dtypes, training_rows=sum(manifest['task_rows'][str(t)] for t in cfg['train_tasks']),
            normalizer_source=str(Path(normalizer).resolve()) if normalizer else 'reconstructed_from_current_training_data',
            historical_task_evidence=history, action_coordinates=action_coordinates, action_evidence=action_evidence,
            action_low=env.action_space.low, action_high=env.action_space.high, core_sha256=bundle['hashes'],
            historical_normalizer_identity='not established by reconstruction; retrain under shared protocol'))
        write_json(temporary / 'validation.json', dict(
            artifact_status='passed', backend=backend, fingerprints='passed', train_only_statistics='passed',
            task_manifest='passed', historical_task_identity=history['status'],
            focal_cross_validation='not_run', gentle_training='not_run', full_training='not_run',
            files=len(manifest['files']), task_rows=manifest['task_rows']))
        if target.exists():
            raise FileExistsError('Export target appeared during preparation: ' + str(target))
        temporary.rename(target)
        temporary = None
        return target, 'exported'
    finally:
        try:
            env.close()
        finally:
            # Failed staging directories are retained for inspection, never published.
            if locked:
                lock.rmdir()


def write_report(root):
    root = Path(root)
    lines = ['# Shared artifact export report', '',
             'Only published bundles with validation.json are listed as exported. No training is implied.', '',
             '| Environment | Status | Files | Train rows | Test rows | Historical tasks |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    details = []
    for name in ENV_NAMES:
        path = root / name
        if (path / 'validation.json').exists():
            v = read_json(path / 'validation.json')
            m = read_json(path / 'dataset_manifest.json')
            tasks = read_json(path / 'task_manifest.json')
            counts = [sum(m['task_rows'][str(t)] for t in tasks[key]) for key in ('train_tasks', 'eval_tasks')]
            lines.append('| {} | {} | {} | {} | {} | {} |'.format(name, v['artifact_status'], v['files'], *counts, v['historical_task_identity']))
            details += ['', '## ' + name, '', 'Path: `' + str(path.resolve()) + '`', '']
            details += ['- `{}`: `{}`'.format(f, file_hash(path / f)) for f in CORE_FILES]
        else:
            fail = root / (name + '.failure.json')
            reason = read_json(fail)['error'].replace('|', '/') if fail.exists() else 'not executed'
            lines.append('| {} | {} | — | — | — | unverified |'.format(name, reason))
    lines += ['', 'FOCAL import/smoke and GENTLE training verification must be reported separately.',
              'Missing historical task records do not establish historical dataset-task identity.'] + details
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='.export-report-',
                                     suffix='.tmp', dir=root, delete=False) as f:
        f.write('\n'.join(lines) + '\n')
        tmp = Path(f.name)
    tmp.replace(root / 'export_report.md')
