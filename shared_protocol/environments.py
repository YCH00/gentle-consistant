# Shared artifact schema ported from the user's FOCAL/focal_shared (2026-10-11).
"""Load the pinned reference environments without executing its placeholder registry.

Only the import namespace is rewritten; rewards, resets, dynamics, XML paths and
task sampling execute directly from the reference checkout. No reference files
are modified. GENTLE's network and optimization implementations remain separate.
"""
import importlib
import importlib.abc
import importlib.util
import hashlib
from pathlib import Path
import sys
import types

import numpy as np

from shared_protocol.common import PROTOCOL, REFERENCE_COMMIT, digest, file_hash, read_json, write_json
from shared_protocol.data import normalize

ALIAS = 'gentle_shared_reference'
MODULES = {'point-robot': ('point_robot', 'PointEnv'),
           'ant-dir': ('ant_dir', 'AntDirEnv'),
           'cheetah-vel': ('half_cheetah_vel', 'HalfCheetahVelEnv'),
           'cheetah-dir': ('half_cheetah_dir', 'HalfCheetahDirEnv'),
           'hopper-rand-params': ('hopper_rand_params_wrapper', 'HopperRandParamsWrappedEnv'),
           'walker-rand-params': ('walker_rand_params_wrapper', 'WalkerRandParamsWrappedEnv')}


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def verify_reference(root):
    root = Path(root).resolve()
    pinned = read_json(Path(__file__).with_name('reference_sources.json'))
    bad = [name for name, sha in pinned['files'].items()
           if not (root / name).is_file() or source_hash(root / name) != sha]
    if bad:
        raise ValueError(f'Reference source differs from {REFERENCE_COMMIT}: {bad[:8]}')
    return root, digest(pinned)


class ReferenceLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, root):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(ALIAS + '.'):
            return None
        relative = fullname[len(ALIAS) + 1:].replace('.', '/')
        filename = self.root / 'rlkit/envs' / (relative + '.py')
        package = False
        if not filename.exists():
            filename = self.root / 'rlkit/envs' / relative / '__init__.py'
            package = True
        if not filename.exists():
            return None
        return importlib.util.spec_from_file_location(fullname, filename, loader=self,
                    submodule_search_locations=[str(filename.parent)] if package else None)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source = Path(module.__spec__.origin).read_text(encoding='utf-8')
        source = source.replace('from rlkit.envs', 'from ' + ALIAS)
        exec(compile(source, module.__spec__.origin, 'exec'), module.__dict__)


def make_env(cfg, reference_root):
    import importlib.metadata
    if importlib.metadata.version('gym') != '0.17.2':
        raise RuntimeError('Shared reference environments require gym==0.17.2; do not substitute a same-name modern environment')
    root, source_fingerprint = verify_reference(reference_root)
    if ALIAS not in sys.modules:
        package = types.ModuleType(ALIAS)
        package.__path__ = [str(root / 'rlkit/envs')]
        package.reference_root = root
        package.register_env = lambda name: lambda cls: cls
        sys.modules[ALIAS] = package
        sys.meta_path.insert(0, ReferenceLoader(root))
    elif sys.modules[ALIAS].reference_root != root:
        raise ValueError('One reference checkout per Python process is required')
    name = cfg['environment']
    if name in ('hopper-rand-params', 'walker-rand-params'):
        if 'rand_param_envs' in sys.modules:
            if Path(sys.modules['rand_param_envs'].__file__).resolve().parent != root / 'rand_param_envs':
                raise ValueError('Wrong rand_param_envs package already imported')
        else:
            spec = importlib.util.spec_from_file_location('rand_param_envs', root / 'rand_param_envs/__init__.py',
                        submodule_search_locations=[str(root / 'rand_param_envs')])
            module = importlib.util.module_from_spec(spec)
            sys.modules['rand_param_envs'] = module
            spec.loader.exec_module(module)
    module, cls = MODULES[name]
    env_type = getattr(importlib.import_module(ALIAS + '.' + module), cls)
    state = np.random.get_state()
    try:
        np.random.seed(1337)
        env = env_type(**cfg['env_params'])
    finally:
        np.random.set_state(state)
    env.shared_backend = dict(class_name=env_type.__name__, source=str(sys.modules[env_type.__module__].__file__),
                              reference_commit=REFERENCE_COMMIT, reference_sources_sha256=source_fingerprint,
                              placeholder=False)
    if name != 'point-robot' and not hasattr(env, 'model'):
        raise RuntimeError('Real MuJoCo model required; placeholder environment rejected')
    if name.startswith('cheetah'):
        import gym
        xml = Path(gym.__file__).parent / 'envs/mujoco/assets/half_cheetah.xml'
    elif name == 'ant-dir':
        xml = root / 'rlkit/envs/mujoco/assets/ant.xml'
    elif 'rand-params' in name:
        xml_name = 'hopper.xml' if name.startswith('hopper') else 'walker2d.xml'
        xml = root / 'rand_param_envs/gym/envs/mujoco/assets' / xml_name
    else:
        xml = None
    env.shared_backend['xml'] = None if xml is None else dict(path=str(xml), sha256=file_hash(xml))
    return env


def task_payload(env, cfg):
    tasks = env.goals if cfg['environment'] == 'point-robot' else env.tasks
    return dict(protocol_id=PROTOCOL, environment=cfg['environment'], task_sampling_seed=1337,
                train_tasks=cfg['train_tasks'], eval_tasks=cfg['eval_tasks'], horizon=cfg['horizon'],
                tasks=tasks, reference_commit=REFERENCE_COMMIT,
                reference_sources_sha256=env.shared_backend['reference_sources_sha256'])


def export_or_check_tasks(env, cfg, path, create=False):
    actual = task_payload(env, cfg)
    if create:
        write_json(path, actual)
    else:
        expected = read_json(path)
        if digest(expected) != digest(actual):
            raise ValueError('Task manifest differs from actual reference environment tasks/split/source')
        # Load exactly the shared arrays after checking the generated task identities.
        if cfg['environment'] == 'point-robot':
            env.goals = expected['tasks']
        else:
            env.tasks = [{key: np.asarray(value) if isinstance(value, list) else value
                          for key, value in task.items()} for task in expected['tasks']]
        env.reset_task(0)
    return actual

