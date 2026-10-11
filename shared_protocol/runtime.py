"""Opt-in artifact consumption for GENTLE; legacy entry points remain unchanged."""
from pathlib import Path
import shutil

import numpy as np

from shared_protocol.common import PROTOCOL, digest, file_hash, read_json, write_json
from shared_protocol.data import load_raw, fit_normalizer, load_normalizer, normalize
from shared_protocol.environments import make_env, export_or_check_tasks

ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = ('task_manifest.json', 'observation_normalizer.npz', 'dataset_manifest.json')
ENV_NAMES = ('point-robot', 'ant-dir', 'cheetah-vel', 'cheetah-dir',
             'hopper-rand-params', 'walker-rand-params')


def protocol_config(name, data_dir):
    if name not in ENV_NAMES:
        raise ValueError('Unsupported shared environment: ' + name)
    point, discrete = name == 'point-robot', name == 'cheetah-dir'
    horizon = 20 if point else 200
    params = dict(n_tasks=2 if discrete else 20, randomize_tasks=name != 'walker-rand-params',
                  max_episode_steps=horizon)
    if name == 'ant-dir':
        params['forward_backward'] = False
    return dict(protocol_id=PROTOCOL, environment=name, task_sampling_seed=1337,
                env_params=params, horizon=horizon,
                train_tasks=list(range(2 if discrete else 10)),
                eval_tasks=list(range(2)) if discrete else list(range(10, 20)),
                n_trj=100 if point or discrete else 50,
                data_checkpoint=100000 if point else 1000000,
                data_dir=str(Path(data_dir).expanduser().resolve()))


def configure(variant, artifacts=None, data_dir=None):
    if artifacts is not None:
        variant['shared_artifacts'] = str(Path(artifacts).expanduser().resolve())
    if data_dir is not None:
        variant['algo_params']['data_dir'] = str(Path(data_dir).expanduser().resolve())
    if not variant.get('shared_artifacts'):
        return
    cfg = config_from_variant(variant)
    path = Path(variant['shared_artifacts'])
    hashes = {name: file_hash(path / name) for name in CORE_FILES}
    previous = variant.get('shared_artifact_hashes')
    if previous and previous != hashes:
        raise ValueError('Shared artifacts differ from recorded run fingerprints')
    variant['shared_artifact_hashes'] = hashes
    return cfg


def config_from_variant(variant):
    a = variant['algo_params']
    cfg = protocol_config(variant['env_name'], a['data_dir'])
    expected = dict(n_trj=cfg['n_trj'], train_epoch=cfg['data_checkpoint'],
                    eval_epoch=cfg['data_checkpoint'], max_path_length=cfg['horizon'], sample=1)
    for key, value in expected.items():
        if a.get(key) != value:
            raise ValueError('Shared mode requires {}={}'.format(key, value))
    if (variant['n_train_tasks'] != len(cfg['train_tasks']) or
            variant['n_eval_tasks'] != len(cfg['eval_tasks'])):
        raise ValueError('Shared mode task split differs from shared-main-v1')
    for key, value in cfg['env_params'].items():
        if variant['env_params'].get(key, value) != value:
            raise ValueError('Shared environment parameter differs: ' + key)
    if set(variant['env_params']) - set(cfg['env_params']):
        raise ValueError('Unexpected shared environment parameters')
    return cfg


def bundle_hashes(path):
    return {name: file_hash(Path(path) / name) for name in CORE_FILES}


def load_bundle(cfg, raw_env, artifacts):
    """Verify raw files and identities, then retain exact published statistics."""
    artifacts = Path(artifacts).expanduser().resolve()
    export_or_check_tasks(raw_env, cfg, artifacts / 'task_manifest.json')
    saved = read_json(artifacts / 'dataset_manifest.json')
    raw, actual = load_raw(cfg, raw_env.observation_space.shape[0], raw_env.action_space.shape[0],
                           saved['data_action_coordinates'])
    actual['normalizer_sha256'] = file_hash(artifacts / 'observation_normalizer.npz')
    actual['task_manifest_sha256'] = file_hash(artifacts / 'task_manifest.json')
    if digest(actual) != digest(saved):
        raise ValueError('Dataset/artifact fingerprint mismatch')
    stats = load_normalizer(artifacts / 'observation_normalizer.npz', fit_normalizer(raw, cfg['train_tasks']))
    with np.load(artifacts / 'observation_normalizer.npz', allow_pickle=False) as f:
        if 'format_version' in f and (f['format_version'].shape != () or f['format_version'].item() != 1):
            raise ValueError('Unsupported shared normalizer format')
        for key in ('train_tasks', 'eval_tasks'):
            if key in f and not np.array_equal(f[key], cfg[key]):
                raise ValueError('Shared normalizer task split mismatch: ' + key)
    # Convert only actions here. Observations stay raw for both training and diagnostics.
    for task, fields in raw.items():
        if saved['data_action_coordinates'] == 'physical':
            low, high = raw_env.action_space.low, raw_env.action_space.high
            fields['actions'] = 2 * (fields['actions'] - low) / (high - low) - 1
        if np.max(np.abs(fields['actions'])) > 1 + 1e-6:
            raise ValueError('Task {}: actions outside declared coordinates'.format(task))
        fields['row_ids'] = np.arange(len(fields['observations']))
    return dict(path=str(artifacts), cfg=cfg, stats=stats, pools=raw,
                manifest=saved, hashes=bundle_hashes(artifacts))


def make_training_env(variant):
    from rlkit.envs.wrappers import NormalizedBoxEnv
    if not variant.get('shared_artifacts'):
        from rlkit.envs import ENVS
        return NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    configure(variant)
    cfg = config_from_variant(variant)
    raw = make_env(cfg, ROOT)
    try:
        bundle = load_bundle(cfg, raw, variant['shared_artifacts'])
        if bundle['hashes'] != variant['shared_artifact_hashes']:
            raise ValueError('Shared bundle changed during loading')
        env = NormalizedBoxEnv(raw)
        env.shared_bundle = bundle
        env.update_obs_mean_var(bundle['stats']['mean'], bundle['stats']['var'])
        return env
    except Exception:
        raw.close()
        raise


def install_normalizer(normalizer, bundle):
    # Never call update: shared statistics are immutable for the run.
    normalizer.mean = bundle['stats']['mean'].copy()
    normalizer.var = bundle['stats']['var'].copy()
    normalizer.count = float(bundle['stats']['count'])


def fill_buffer(buffer, tasks, bundle):
    for task in tasks:
        p = bundle['pools'][task]
        obs = normalize(p['observations'], bundle['stats'])
        nxt = normalize(p['next_observations'], bundle['stats'])
        for o, a, r, n, terminal in zip(obs, p['actions'], p['rewards'], nxt, p['terminals']):
            buffer.add_sample(task, o, a, r, terminal, n, env_info={})


def persist_bundle(bundle, destination):
    """Copy core bytes unchanged, reject conflicting historical artifacts."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in CORE_FILES:
        src, dst = Path(bundle['path']) / name, destination / name
        if file_hash(src) != bundle['hashes'][name]:
            raise ValueError('Published bundle changed: ' + name)
        if dst.exists() and file_hash(dst) != bundle['hashes'][name]:
            raise FileExistsError('Refusing to overwrite different run artifact: ' + str(dst))
        if not dst.exists():
            shutil.copyfile(src, dst)
    write_json(destination / 'shared_artifact_usage.json', dict(
        protocol_id=PROTOCOL, source=bundle['path'], hashes=bundle['hashes']))


def pretrained_dir(kind, variant, seed):
    base = ROOT / kind / variant['env_name']
    if variant.get('shared_artifacts'):
        configure(variant)
        base /= 'shared-' + variant['shared_artifact_hashes']['dataset_manifest.json'][:16]
    return base / ('expert_seed{}'.format(seed))


def check_pretrained(path, variant):
    if variant.get('shared_artifacts'):
        if bundle_hashes(path) != variant['shared_artifact_hashes']:
            raise ValueError('Pretrained model uses a different shared bundle: ' + str(path))


def prepare_pretrain_output(kind, variant, seed):
    path = pretrained_dir(kind, variant, seed)
    if variant.get('shared_artifacts') and path.exists():
        raise FileExistsError('Shared pretraining output already exists; refusing overwrite: ' + str(path))
    return path


def bind_evaluation_run(variant, log_dir, override=None):
    """Moved runs may use their byte-identical local bundle; legacy runs cannot opt in."""
    if override and not variant.get('shared_artifact_hashes'):
        raise ValueError('Cannot replace a legacy checkpoint transform with a shared bundle')
    if variant.get('shared_artifact_hashes'):
        variant['shared_artifacts'] = str(Path(override or log_dir).resolve())
        configure(variant)
