"""Shared diagnostics invariants; does not import a training project."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     default=lambda x: x.tolist(), allow_nan=False).encode()).hexdigest()


def seed32(protocol, environment, task, purpose, condition='shared', repeat=0, rollout=0):
    key = [protocol, environment, int(task), purpose, condition, int(repeat), int(rollout)]
    return int.from_bytes(hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).digest()[:4], 'big')


@contextmanager
def isolated_rng(seed):
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.dtype.str, array.shape)).encode() + array.tobytes()).hexdigest()


def context_hash(context):
    return digest({key: array_hash(value) for key, value in sorted(context.items())})


def weights_hash(modules):
    return digest({name: {key: array_hash(value.detach().cpu().numpy()) for key, value in module.state_dict().items()}
                   for name, module in sorted(modules.items())})


def validate_context(context, obs_dim, action_dim, length=200):
    widths = {'observations': obs_dim, 'actions': action_dim, 'rewards': 1,
              'next_observations': obs_dim, 'episode_end': 1}
    for key, width in widths.items():
        value = np.asarray(context[key])
        if value.shape != (length, width) or not np.isfinite(value).all():
            raise ValueError('Invalid context field: ' + key)
    if np.max(np.abs(context['actions'])) > 1 + 1e-6:
        raise ValueError('Bank actions are not normalized')
    if not np.isin(context['episode_end'], [0, 1]).all():
        raise ValueError('Invalid episode boundary feature')


def load_stats(path):
    with np.load(path, allow_pickle=False) as src:
        stats = {key: src[key].copy() for key in ('mean', 'var', 'count')}
    if (stats['mean'].ndim != 1 or stats['mean'].shape != stats['var'].shape
            or stats['count'].shape != () or stats['count'] <= 0
            or any(not np.isfinite(v).all() for v in stats.values()) or (stats['var'] < 0).any()):
        raise ValueError('Invalid frozen observation statistics')
    return stats


def native_rms(observations):
    """Reconstruct the historical single training update; never use test/context data."""
    mean = np.mean(observations, axis=0)
    var = np.var(observations, axis=0)
    n = len(observations)
    count = n + 1e-4
    delta = mean - np.zeros(observations.shape[1], dtype=np.float64)
    return dict(mean=delta * n / count,
                var=(np.ones_like(delta) * 1e-4 + var * n + delta ** 2 * 1e-4 * n / count) / count,
                count=np.asarray(count))


def normalize(values, stats):
    return values if stats is None else (values - stats['mean']) / np.sqrt(stats['var'] + 1e-8)


def tanh_gaussian_mean(mean, variance):
    """Deterministic quadrature of E[tanh(X)], not the incorrect tanh(E[X])."""
    outputs = []
    for order in (128, 256):
        nodes, weights = np.polynomial.hermite.hermgauss(order)
        nodes = torch.as_tensor(nodes, dtype=torch.float64, device=mean.device)
        weights = torch.as_tensor(weights / np.sqrt(np.pi), dtype=torch.float64, device=mean.device)
        values = mean.double().unsqueeze(-1) + (2 * variance.double()).sqrt().unsqueeze(-1) * nodes
        outputs.append((torch.tanh(values) * weights).sum(-1))
    error = float((outputs[0] - outputs[1]).abs().max())
    if error > 1e-4:
        raise ValueError('Posterior mean quadrature did not converge: ' + str(error))
    return outputs[-1].to(mean.dtype), error
