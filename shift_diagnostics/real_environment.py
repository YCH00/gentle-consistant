"""Pinned legacy MuJoCo environments with tasks restored from the shared manifest."""
import copy
import importlib
from importlib import metadata
import os
from pathlib import Path
import sys
import types

import numpy as np

from audit import sha256
from runtime_common import isolated_rng


def configure_backend(environment, explicit=None):
    version = 131 if environment == 'walker-rand-params' else 150
    choices = [explicit, os.environ.get('MUJOCO_PY_MJPRO_PATH'), str(Path.home() / '.mujoco' / ('mjpro' + str(version)))]
    selected = next((Path(p).expanduser().resolve() for p in choices if p and Path(p).expanduser().is_dir()
                     and Path(p).name == 'mjpro' + str(version)), None)
    if selected is None:
        raise RuntimeError('Existing mjpro{} directory missing; pass --mujoco{} path'.format(version, version))
    os.environ['MUJOCO_PY_MJPRO_PATH'] = str(selected)
    os.environ['LD_LIBRARY_PATH'] = str(selected / 'bin') + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
    return str(selected)


def make_environment(plan, environment):
    # These two helper modules never import their method's policy or training loop.
    for root in (plan['sources']['FOCAL'], plan['sources']['ER-TRL']):
        if root not in sys.path:
            sys.path.append(root)
    helper = importlib.import_module('focal_v2.environments')
    backends = importlib.import_module('ertrl_v2.backends')
    if metadata.version('gym') != '0.17.2':
        raise RuntimeError('Expected main-experiment Gym 0.17.2')
    selected = backends.check_backend_environment(environment)
    root, fingerprint = helper.verify_reference(plan['reference_root'])
    if helper.ALIAS not in sys.modules:
        package = types.ModuleType(helper.ALIAS)
        package.__path__ = [str(root / 'rlkit/envs')]
        package.reference_root = root
        package.register_env = lambda name: lambda cls: cls
        sys.modules[helper.ALIAS] = package
        sys.meta_path.insert(0, helper.ReferenceLoader(root))
    if environment == 'walker-rand-params' and 'rand_param_envs' not in sys.modules:
        spec = importlib.util.spec_from_file_location('rand_param_envs', root / 'rand_param_envs/__init__.py',
                         submodule_search_locations=[str(root / 'rand_param_envs')])
        module = importlib.util.module_from_spec(spec)
        sys.modules['rand_param_envs'] = module
        spec.loader.exec_module(module)
    task_manifest = plan['shared'][environment]['task_payload']
    if task_manifest['reference_sources_sha256'] != fingerprint:
        raise ValueError('Reference implementation differs from task manifest')
    tasks = [{key: np.asarray(value) if isinstance(value, list) else value for key, value in task.items()}
             for task in task_manifest['tasks']]
    module, cls = helper.MODULES[environment]
    native_class = getattr(importlib.import_module(helper.ALIAS + '.' + module), cls)
    class ManifestEnvironment(native_class):
        def sample_tasks(self, num_tasks):
            if num_tasks != len(tasks):
                raise ValueError('Task count mismatch')
            return copy.deepcopy(tasks)
    with isolated_rng(1337):
        env = ManifestEnvironment(n_tasks=len(tasks), randomize_tasks=(environment == 'cheetah-vel'),
                                  max_episode_steps=200)
    if not hasattr(env, 'model'):
        raise RuntimeError('Real MuJoCo model required')
    provenance = plan['shared'][environment]['preparation_payload']
    if environment == 'cheetah-vel':
        import gym
        xml = Path(gym.__file__).parent / 'envs/mujoco/assets/half_cheetah.xml'
    else:
        xml = root / 'rand_param_envs/gym/envs/mujoco/assets/walker2d.xml'
    if sha256(xml) != provenance['backend']['xml']['sha256']:
        raise ValueError('Environment XML differs from main experiment')
    receipt = backends.verify_loaded_backend(environment, selected)
    receipt.update(reference_sources_sha256=fingerprint, xml_path=str(xml), xml_sha256=sha256(xml),
                   task_manifest_sha256=plan['shared'][environment]['task_manifest.json']['sha256'],
                   task_source='manifest supplied directly to constructor; no task sampling')
    return env, receipt


def reset_task(env, task, seed):
    env.seed(seed)
    env.reset_task(task)  # Native reset_task already performs exactly one reset.
    observation = np.asarray(env._get_obs()).copy()
    if not np.isfinite(observation).all():
        raise ValueError('Non-finite reset observation')
    return observation


def step(env, action):
    action = np.asarray(action)
    if action.shape != env.action_space.shape or not np.isfinite(action).all() or np.max(np.abs(action)) > 1 + 1e-6:
        raise ValueError('Invalid normalized action')
    low, high = env.action_space.low, env.action_space.high
    physical = low + (np.clip(action, -1, 1) + 1) * 0.5 * (high - low)
    observation, reward, done, info = env.step(physical)
    observation, reward = np.asarray(observation).copy(), float(reward)
    if not np.isfinite(observation).all() or not np.isfinite(reward):
        raise ValueError('Non-finite environment feedback')
    return observation, reward, bool(done), info
