"""Fixed-context diagnostics, independent of Gym and the training algorithm.

These helpers evaluate a *supplied* context exactly once. They do not infer a
new latent along the rollout, resample context rows, or modify Agent.context.
Q errors use the observed finite rollout tail with zero terminal bootstrap;
they are not ground truth for a continuing, stochastic, or smoothed policy.
"""

from contextlib import contextmanager
import math
import random

import numpy as np
import torch
from torch.nn import functional as F


@contextmanager
def isolated_evaluation(*modules, seed=None):
    """Restore global RNGs and every submodule's original training flag.

The caller owns the environment: its simulator state and private RNG are
intentionally advanced. Use a dedicated evaluation environment.
"""
    if len(modules) == 1 and isinstance(modules[0], (tuple, list)):
        modules = tuple(modules[0])
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    flags = {}
    for module in modules:
        if module is not None:
            for child in module.modules():
                flags.setdefault(child, child.training)
    try:
        for module in modules:
            if module is not None:
                module.eval()
        if seed is not None:
            _seed_globals(seed)
        with torch.no_grad():
            yield
    finally:
        # Assign individually: train() recursively overwrites mixed child modes.
        for module, flag in flags.items():
            module.training = flag
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError('{} must be a positive integer'.format(name))
    return int(value)


def _finite_tensor(value, name):
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise ValueError('{} must be a floating-point tensor'.format(name))
    if not torch.isfinite(value).all().item():
        raise ValueError('{} contains nonfinite values'.format(name))
    return value


def _infer_embedding(encoder, context, latent_dim, use_information_bottleneck):
    _finite_tensor(context, 'context')
    if context.ndim != 3 or context.shape[0] != 1 or min(context.shape[1:]) < 1:
        raise ValueError('context must have shape [1, N, D] with N and D positive')
    latent_dim = _positive_integer(latent_dim, 'latent_dim')
    width = latent_dim * (2 if use_information_bottleneck else 1)
    params = _finite_tensor(encoder(context), 'encoder output')
    if params.ndim not in (2, 3) or params.shape[-1] != width or params.numel() < width:
        raise ValueError('encoder output must contain mean/variance width {}'.format(width))
    params = params.reshape(1, -1, width)
    if use_information_bottleneck:
        mus = params[..., :latent_dim]
        variances = F.softplus(params[..., latent_dim:]).clamp(min=1e-7)
        variance = 1.0 / torch.reciprocal(variances).sum(dim=1)
        latent = variance * (mus / variances).sum(dim=1)
    else:
        latent = params.mean(dim=1)
    return _finite_tensor(latent, 'latent embedding').detach().clone()


def infer_fixed_embedding(encoder, context, latent_dim, use_information_bottleneck=False):
    """Return Agent's posterior *mean* [1, Z], including its IB precision pooling.

IB evaluation deliberately uses the mean instead of drawing a posterior sample.
The supplied context rows are used once, in their original order.
"""
    with isolated_evaluation(encoder):
        return _infer_embedding(encoder, context, latent_dim, use_information_bottleneck)


def _batch_tensor(value, name, reference):
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    _finite_tensor(value, name)
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError('{} must have shape [B, D] with positive dimensions'.format(name))
    return value


def _numpy(value):
    return value.detach().cpu().numpy().copy()


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), std=float(values.std()),
                min=float(values.min()), max=float(values.max()))


def _errors(prediction, target):
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return dict(bias=float(delta.mean()), mae=float(np.abs(delta).mean()),
                rmse=float(np.sqrt(np.square(delta).mean())))


def diagnose_fixed_batch(decoder, qf1, qf2, observations, actions, rewards,
                         next_observations, z, use_next_obs_in_context=False):
    """Evaluate decoder and Q on one exact supplied transition batch.

Returns ``(JSON-safe summary, numpy arrays)``. Reward targets are unscaled
environment/context rewards; dynamics targets are absolute next observations,
matching GENTLE's decoder. Offline Q statistics have no MC error field: the
behavior trajectories in a replay buffer do not estimate the current policy Q.
"""
    _finite_tensor(z, 'z')
    if z.ndim != 2 or z.shape[0] != 1 or z.shape[1] < 1:
        raise ValueError('z must have shape [1, Z]')
    obs = _batch_tensor(observations, 'observations', z)
    act = _batch_tensor(actions, 'actions', z)
    nxt = _batch_tensor(next_observations, 'next_observations', z)
    reward = torch.as_tensor(rewards, device=z.device, dtype=z.dtype)
    if reward.ndim == 1:
        reward = reward[:, None]
    reward = _batch_tensor(reward, 'rewards', z)
    count = obs.shape[0]
    if act.shape[0] != count or nxt.shape != obs.shape or reward.shape != (count, 1):
        raise ValueError('transition shapes must agree, with rewards [B, 1]')
    repeated_z = z.expand(count, -1)
    arrays = dict(observations=_numpy(obs), actions=_numpy(act), rewards=_numpy(reward),
                  next_observations=_numpy(nxt))
    summary = dict(count=int(count), reward=_distribution(arrays['rewards']))
    with isolated_evaluation(decoder, qf1, qf2):
        if decoder is not None:
            prediction = _finite_tensor(decoder(obs, act, repeated_z), 'decoder output')
            expected_width = 1 + (obs.shape[1] if use_next_obs_in_context else 0)
            if prediction.shape != (count, expected_width):
                raise ValueError('decoder output must have shape [B, {}]'.format(expected_width))
            arrays['decoder_predictions'] = _numpy(prediction)
            summary['decoder_reward_error'] = _errors(arrays['decoder_predictions'][:, :1], arrays['rewards'])
            if use_next_obs_in_context:
                summary['decoder_next_observation_error'] = _errors(arrays['decoder_predictions'][:, 1:], arrays['next_observations'])
        for name, critic in (('q1', qf1), ('q2', qf2)):
            if critic is None:
                continue
            prediction = _finite_tensor(critic(1, count, obs, act, repeated_z), name)
            if prediction.shape != (count, 1):
                raise ValueError('{} output must have shape [B, 1]'.format(name))
            arrays[name] = _numpy(prediction)
            summary[name] = _distribution(arrays[name])
        if 'q1' in arrays and 'q2' in arrays:
            arrays['q_min'] = np.minimum(arrays['q1'], arrays['q2'])
            summary['q_min'] = _distribution(arrays['q_min'])
            summary['q_disagreement'] = float(np.abs(arrays['q1'] - arrays['q2']).mean())
    return summary, arrays


def _observation(value):
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        value = value[0]
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError('environment observations must be finite, nonempty 1D arrays')
    return value.copy()


def _seed_globals(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2 ** 32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def reset_for_task(env, task_idx, seed):
    """Set one task and obtain one explicitly reset observation with fixed RNGs."""
    _seed_globals(seed)
    legacy_seed = callable(getattr(env, 'seed', None))
    if legacy_seed:
        env.seed(seed)
    env.reset_task(task_idx)
    # Existing reset_task methods may internally reset but return no observation.
    # One explicit reset is required to obtain the wrapped/normalized observation.
    return _observation(env.reset() if legacy_seed else env.reset(seed=seed))


def step_environment(env, action):
    """Normalize old/new Gym step formats without conflating timeout and death."""
    result = env.step(np.asarray(action).copy())
    if not isinstance(result, (tuple, list)) or len(result) not in (4, 5):
        raise ValueError('env.step must return a Gym 4-tuple or Gymnasium 5-tuple')
    if len(result) == 5:
        nxt, reward, terminal, timeout, info = result
        terminal, timeout = bool(terminal), bool(timeout)
    else:
        nxt, reward, done, info = result
        if not isinstance(info, dict):
            raise ValueError('env.step info must be a dictionary')
        timeout = bool(done) and bool(info.get('TimeLimit.truncated', False))
        terminal = bool(done) and not timeout
    if not isinstance(info, dict):
        raise ValueError('env.step info must be a dictionary')
    info = dict(info)
    # Old project environments also use done=True for their internal time limit.
    # Preserve the environment's flag, but do not claim it proves true termination.
    info['_fixed_context_legacy_done_ambiguous'] = (
        len(result) == 4 and terminal and 'TimeLimit.truncated' not in info)
    reward_array = np.asarray(reward)
    if reward_array.size != 1 or not np.isfinite(reward_array).all():
        raise ValueError('environment reward must be one finite scalar')
    return _observation(nxt), float(reward_array.reshape(-1)[0]), terminal, timeout, info


def _policy_action(policy, obs, z, deterministic):
    obs_tensor = torch.as_tensor(obs, device=z.device, dtype=z.dtype)[None, :]
    output = policy(1, 1, torch.cat((obs_tensor, z), dim=-1), deterministic=deterministic)
    action = output[0] if isinstance(output, (tuple, list)) else output
    _finite_tensor(action, 'policy action')
    if action.ndim != 2 or action.shape[0] != 1 or action.shape[1] < 1:
        raise ValueError('policy action must have shape [1, A]')
    return _numpy(action)[0]


def collect_zero_context(env, policy, task_idx, seed, context_size, max_path_length,
                         latent_dim):
    """Collect exactly N fresh transitions with the stochastic actor at z=0.

    The bank uses environment rewards (dense contexts). It contains each observed
    transition once; no replay resampling or repeated-context accumulation occurs.
    A collection boundary is recorded as a timeout if it cuts an episode short.
    """
    count = _positive_integer(context_size, 'context_size')
    horizon = _positive_integer(max_path_length, 'max_path_length')
    dimension = _positive_integer(latent_dim, 'latent_dim')
    parameter = next(policy.parameters(), None)
    device = parameter.device if parameter is not None else torch.device('cpu')
    dtype = parameter.dtype if parameter is not None else torch.float32
    z = torch.zeros(1, dimension, device=device, dtype=dtype)
    records = {key: [] for key in ('observations', 'actions', 'rewards',
               'next_observations', 'terminals', 'timeouts')}
    with isolated_evaluation(policy):
        obs = reset_for_task(env, task_idx, seed)
        episode, step = 0, 0
        for index in range(count):
            action = _policy_action(policy, obs, z, deterministic=False)
            nxt, reward, terminal, timeout, _ = step_environment(env, action)
            if nxt.shape != obs.shape:
                raise ValueError('observation shape changed within rollout')
            step += 1
            timeout = timeout or ((step == horizon or index + 1 == count) and not terminal)
            for key, value in (('observations', obs), ('actions', action),
                               ('rewards', [reward]), ('next_observations', nxt),
                               ('terminals', [terminal]), ('timeouts', [timeout])):
                records[key].append(value)
            obs = nxt
            if (terminal or timeout) and index + 1 < count:
                episode, step = episode + 1, 0
                obs = reset_for_task(env, task_idx, (int(seed) + episode) % (2 ** 32))
    return {key: np.asarray(value, dtype=np.bool_ if key in ('terminals', 'timeouts') else np.float32)
            for key, value in records.items()}


def _tail_returns(rewards, discount):
    result = np.empty(len(rewards), dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + discount * running
        result[index] = running
    return result


def evaluate_fixed_context(env, policy, encoder, context, task_idx, reset_seeds,
                           max_path_length, discount, reward_scale, latent_dim,
                           use_information_bottleneck=False, qf1=None, qf2=None,
                           decoder=None, use_next_obs_in_context=False):
    """Evaluate a paired encoder/actor under one frozen context and frozen z.

Returns ``(summary, trajectories)``. Summary is JSON-safe; trajectories contain
NumPy arrays. ``reward_scale`` is the training critic's scale, applied exactly
once when forming MC targets. The environment is expected to return the same
reward units as replay/context. Actions use the deterministic actor, without
exploration noise or target-policy smoothing.
"""
    horizon = _positive_integer(max_path_length, 'max_path_length')
    for name, value in (('discount', discount), ('reward_scale', reward_scale)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('{} must be finite'.format(name))
    if not 0 <= discount <= 1 or reward_scale <= 0:
        raise ValueError('discount must be in [0, 1] and reward_scale positive')
    seeds = list(reset_seeds)
    if not seeds:
        raise ValueError('reset_seeds must be nonempty')
    for seed in seeds:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2 ** 32:
            raise ValueError('reset_seeds must contain integers in [0, 2**32)')
    seeds = [int(seed) for seed in seeds]
    trajectories = []
    per_rollout = []
    with isolated_evaluation(policy, encoder, decoder, qf1, qf2):
        z = _infer_embedding(encoder, context, latent_dim, use_information_bottleneck)
        for seed in seeds:
            obs = reset_for_task(env, task_idx, seed)
            records = {key: [] for key in ('observations', 'actions', 'rewards',
                       'next_observations', 'terminals', 'timeouts')}
            for step in range(horizon):
                action = _policy_action(policy, obs, z, deterministic=True)
                nxt, reward, terminal, timeout, final_info = step_environment(env, action)
                if nxt.shape != obs.shape:
                    raise ValueError('observation shape changed within rollout')
                timeout = timeout or (step + 1 == horizon and not terminal)
                for key, value in (('observations', obs), ('actions', action),
                                   ('rewards', [reward]), ('next_observations', nxt),
                                   ('terminals', [terminal]), ('timeouts', [timeout])):
                    records[key].append(value)
                obs = nxt
                if terminal or timeout:
                    break
            trajectory = {key: np.asarray(value, dtype=np.bool_ if key in ('terminals', 'timeouts') else np.float32)
                          for key, value in records.items()}
            raw_tail = _tail_returns(trajectory['rewards'][:, 0], discount)
            trajectory['discounted_raw_returns'] = raw_tail[:, None]
            trajectory['discounted_scaled_returns'] = reward_scale * raw_tail[:, None]
            diagnostics, predictions = diagnose_fixed_batch(
                decoder, qf1, qf2, trajectory['observations'], trajectory['actions'],
                trajectory['rewards'], trajectory['next_observations'], z, use_next_obs_in_context)
            for key in ('decoder_predictions', 'q1', 'q2', 'q_min'):
                if key in predictions:
                    trajectory[key] = predictions[key]
                if key.startswith('q') and key in predictions:
                    diagnostics[key + '_finite_horizon_mc_error'] = _errors(
                        predictions[key], trajectory['discounted_scaled_returns'])
            per_rollout.append(dict(reset_seed=seed, length=len(raw_tail),
                return_raw=float(trajectory['rewards'].astype(np.float64).sum()),
                discounted_return_raw=float(raw_tail[0]),
                discounted_return_scaled=float(reward_scale * raw_tail[0]),
                terminated=bool(trajectory['terminals'][-1, 0]),
                truncated=bool(trajectory['timeouts'][-1, 0]), horizon_reached=len(raw_tail) == horizon,
                termination_cause_ambiguous=bool(final_info['_fixed_context_legacy_done_ambiguous']),
                diagnostics=diagnostics))
            trajectories.append(trajectory)
    returns = [item['return_raw'] for item in per_rollout]
    summary = dict(task_idx=int(task_idx), context_size=int(context.shape[1]),
        z=_numpy(z).tolist(), use_posterior_mean=True, deterministic_policy=True,
        reset_seeds=seeds, returns=returns, return_mean=float(np.mean(returns)),
        return_std=float(np.std(returns)), discount=float(discount), reward_scale=float(reward_scale),
        max_path_length=horizon, per_rollout=per_rollout,
        mc_target_note='Observed finite rollout tail, zero bootstrap; not ground truth for continuing or smoothed-policy Q.',
        termination_note='terminated preserves environment-reported flags. Legacy done without truncation metadata has ambiguous cause, including internal time limits.')
    # Pool transitions, rather than average per-rollout errors with unequal lengths.
    for key in ('q1', 'q2', 'q_min'):
        if key in trajectories[0]:
            predictions = np.concatenate([item[key] for item in trajectories])
            targets = np.concatenate([item['discounted_scaled_returns'] for item in trajectories])
            summary[key] = _distribution(predictions)
            summary[key + '_finite_horizon_mc_error'] = _errors(predictions, targets)
    if decoder is not None:
        predictions = np.concatenate([item['decoder_predictions'] for item in trajectories])
        rewards = np.concatenate([item['rewards'] for item in trajectories])
        summary['decoder_reward_error'] = _errors(predictions[:, :1], rewards)
        if use_next_obs_in_context:
            next_obs = np.concatenate([item['next_observations'] for item in trajectories])
            summary['decoder_next_observation_error'] = _errors(predictions[:, 1:], next_obs)
    return summary, trajectories
