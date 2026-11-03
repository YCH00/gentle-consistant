"""Inference-only adapters for the exact source layouts verified in the evidence bundle.

Each process imports ONE method checkout. No optimizers, training algorithms,
replay buffers or method-specific context collectors are constructed here.
"""
import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from audit import sha256
from runtime_common import digest, normalize, tanh_gaussian_mean, validate_context


def load_local_checkpoint(path, expected_hash):
    path = Path(path)
    if sha256(path) != expected_hash:
        raise ValueError('Checkpoint changed after selection: ' + str(path))
    # User-owned local training checkpoints contain NumPy/RNG objects. Exact
    # path/hash is bound before deserialization; never load arbitrary downloads.
    return torch.load(str(path), map_location='cpu', weights_only=False)


def artifact_binding(binding, shared):
    values = binding.get('source_artifact_hashes', binding.get('artifact_hashes', {}))
    for name in ('task_manifest.json', 'dataset_manifest.json', 'observation_normalizer.npz'):
        if values.get(name) != shared[name]['sha256']:
            raise ValueError('Checkpoint shared-artifact binding differs: ' + name)


def verify_implementation(files, root, normalize_lf=False):
    import hashlib
    for name, expected in files.items():
        path = (Path(root) / name).resolve()
        if Path(root).resolve() not in path.parents:
            raise ValueError('Unsafe source binding')
        raw = path.read_bytes()
        if normalize_lf:
            raw = raw.replace(b'\r\n', b'\n')
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Training source changed: ' + name)


def policy_state(agent_state):
    prefix = 'policy.'
    result = {k[len(prefix):]: v for k, v in agent_state.items() if k.startswith(prefix)}
    if not result:
        raise ValueError('Native agent has no policy weights')
    return result


def native_source_digest(method, root):
    """Same file lists and raw-byte hashes as the native checkpoint writers."""
    root = Path(root)
    if method == 'CSRO':
        paths = [p for pattern in ('rlkit/**/*.py', 'csro_shared/*.py', 'scripts/*.py',
                                   'configs/shared_protocol/*.json') for p in root.glob(pattern)]
        paths.append(root / 'csro_shared/reference_sources.json')
    elif method == 'CORRO':
        paths = [p for p in (root / 'shared_native').glob('*') if p.suffix in ('.py', '.json')]
        paths += [root / p for p in ('run_shared_native.py', 'algorithms/sac.py', 'models/encoder.py',
                                    'models/generative.py', 'models/policy.py', 'utils/helpers.py')]
        paths += list((root / 'torchkit').glob('*.py'))
    else:
        paths = list((root / 'unicorn_v2').glob('*.py'))
        paths += [root / p for p in ('models/encoder.py', 'models/decoder.py', 'models/policy.py',
            'algorithms/sac.py', 'algorithms/focalsac.py', 'torchkit/networks.py', 'torchkit/pytorch_utils.py',
            'torchkit/brac/divergences.py', 'torchkit/brac/utils.py', 'torchkit/distributions.py',
            'torchkit/core.py', 'torchkit/modules.py', 'unicorn_v2/reference_sources.json',
            'unicorn_v2/reference_dependencies.json', 'run_shared_native_v2.py', 'launch_shared_native_v2.py')]
    return digest({p.relative_to(root).as_posix(): sha256(p) for p in paths})


class NativeAdapter:
    def __init__(self, entry, shared, obs_dim, action_dim, device='cpu', stats=None):
        self.entry, self.method, self.stats = entry, entry['method'], stats
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.device = torch.device(device)
        self.report = {'strict_loading': True, 'preprocessing': 'identity' if stats is None else 'native_RMS_once'}
        self.cfg = entry['configuration']
        sys.path.insert(0, entry['source_root'])
        self.modules, self.agent = {}, None
        paths = entry['checkpoint_files']
        states = [load_local_checkpoint(x['path'], x['sha256']) for x in paths]
        if self.method in ('GENTLE', 'Ours-local', 'Ours-global', 'FOCAL', 'CSRO', 'ER-TRL'):
            self._rlkit(states, shared)
        else:
            self._transition(states[0], shared)
        for module in self.modules.values():
            module.to(self.device).eval().requires_grad_(False)
        if self.agent is not None:
            self.agent.to(self.device).eval().requires_grad_(False)
        self.report.update(latent_dim=self.latent_dim, native_input_fields=self.fields,
                           latent_kind='posterior_mean_quadrature' if self.method == 'UNICORN-SS' else 'native_deterministic')

    def tensor(self, array):
        return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=self.device)

    def _rlkit(self, states, shared):
        method, cfg = self.method, self.cfg
        legacy = method in ('GENTLE', 'Ours-local', 'Ours-global')
        native = cfg if legacy or method == 'CSRO' else cfg['native']
        a = native['algo_params']
        if a.get('use_information_bottleneck') or a.get('recurrent') or a.get('sparse_rewards'):
            raise ValueError('This adapter has only verified deterministic, dense-reward MLP native runs')
        ptu = importlib.import_module('rlkit.torch.pytorch_util')
        ptu.device = self.device
        ptu._use_gpu, ptu._gpu_id = self.device.type == 'cuda', self.device.index or 0
        if legacy:
            encoder_cls = importlib.import_module('rlkit.torch.autoencoder').MlpEncoder
            agent_cls = importlib.import_module('rlkit.torch.sac.agent').Agent
        else:
            encoder_cls = importlib.import_module('rlkit.torch.networks').MlpEncoder
            agent_mod = 'rlkit.torch.agents.agent' if method == 'ER-TRL' else 'rlkit.torch.sac.agent'
            agent_cls = importlib.import_module(agent_mod).PEARLAgent
        policy_mod = 'rlkit.torch.agents.policies' if method == 'ER-TRL' else 'rlkit.torch.sac.policies'
        policy_cls = importlib.import_module(policy_mod).TanhGaussianPolicy
        self.latent_dim = native['latent_size']
        self.fields = ['observations', 'actions', 'rewards'] + (['next_observations'] if a['use_next_obs_in_context'] else [])
        inputs = self.obs_dim + self.action_dim + 1 + (self.obs_dim if a['use_next_obs_in_context'] else 0)
        hidden = [200] * 3 if method == 'FOCAL' else native['encoder_dims'] if method == 'ER-TRL' else [native['net_size']] * 3
        kwargs = {'layer_norm': a.get('layer_norm', False)} if method == 'CSRO' else {}
        encoder = encoder_cls(hidden_sizes=hidden, input_size=inputs, output_size=self.latent_dim,
                              output_activation=torch.tanh, **kwargs)
        policy = policy_cls(hidden_sizes=[native['net_size']] * 3, obs_dim=self.obs_dim + self.latent_dim,
                            latent_dim=self.latent_dim, action_dim=self.action_dim)
        self.agent = agent_cls(self.latent_dim, encoder, policy, **a)
        if legacy:
            if len(states) != 2:
                raise ValueError('Legacy encoder/policy pair required')
            enc_state, pol_state = states
            self.report['provenance_limit'] = 'Legacy state_dict files lack embedded training source/seed metadata; user-selected same-run pair, configuration and data provenance are recorded separately'
        elif method == 'CSRO':
            state = states[0]
            if state['method'] != 'CSRO' or state['test_only'] or state['training_seed'] != self.entry['training_seed'] or state['iteration'] != 499:
                raise ValueError('CSRO checkpoint identity/iteration invalid')
            if state['scientific_config'] != importlib.import_module('csro_shared.configuration').scientific_config(cfg):
                raise ValueError('CSRO training configuration mismatch')
            artifact_binding(state, shared)
            if state['source_fingerprint'] != native_source_digest(method, self.entry['source_root']):
                raise ValueError('CSRO training source fingerprint mismatch')
            if state['main_updates'] != 500 * cfg['algo_params']['num_train_steps_per_itr']:
                raise ValueError('CSRO incomplete training counters')
            enc_state, pol_state = state['networks']['context_encoder'], state['networks']['policy']
        else:
            state, binding = states[0], states[0]['binding']
            if binding['config_hash'] != digest(cfg) or binding['training_seed'] != self.entry['training_seed']:
                raise ValueError('Native checkpoint config/seed mismatch')
            if state['smoke'] or not state['completed_native_plan'] or state['next_iteration'] - 1 != self.entry['final_iteration_zero_based']:
                raise ValueError('Incomplete native checkpoint')
            plan_module = 'focal_v2.config' if method == 'FOCAL' else 'ertrl_v2.config'
            native_plan = importlib.import_module(plan_module).training_plan(cfg)
            expected_counts = native_plan['optimizer_steps' if method == 'FOCAL' else 'expected_optimizer_steps']
            if (binding['training_plan_hash'] != digest(native_plan) or state['main_updates'] != native_plan['main_updates']
                    or state['meta_updates'] != a['num_iterations'] * a['num_train_steps_per_itr']
                    or state['optimizer_counts'] != expected_counts):
                raise ValueError('Native final training budget/counters mismatch')
            artifact_binding(binding, shared)
            verify_implementation(binding['implementation_hashes'], self.entry['source_root'], normalize_lf=True)
            enc_state, pol_state = state['encoder'], state['policy']
        encoder.load_state_dict(enc_state, strict=True)
        policy.load_state_dict(pol_state, strict=True)
        self.modules = {'encoder': encoder, 'policy': policy}

    def _transition(self, state, shared):
        cfg = self.cfg
        encoder_cls = importlib.import_module('models.encoder').MLPEncoder
        policy_cls = importlib.import_module('models.policy').TanhGaussianPolicy
        ptu = importlib.import_module('torchkit.pytorch_utils')
        ptu.device = self.device
        ptu._use_gpu, ptu._gpu_id = self.device.type == 'cuda', self.device.index or 0
        unicorn = self.method == 'UNICORN-SS'
        a = cfg['native'] if unicorn else cfg
        self.latent_dim = a['task_embedding_size']
        self.fields = ['observations', 'actions', 'rewards', 'next_observations'] + (['episode_end'] if unicorn else [])
        kwargs = {'stochasity': True} if unicorn else {'normalize': a['normalize_z']}
        encoder = encoder_cls(hidden_size=a['aggregator_hidden_size'], num_hidden_layers=2,
                              task_embedding_size=self.latent_dim, action_size=self.action_dim,
                              state_size=self.obs_dim, reward_size=1, term_size=1 if unicorn else 0, **kwargs)
        policy = policy_cls(obs_dim=self.obs_dim + self.latent_dim, action_dim=self.action_dim,
                            hidden_sizes=a['policy_layers'])
        identity = state['identity']
        if identity['config_hash'] != digest(cfg):
            raise ValueError('Transition-encoder checkpoint config mismatch')
        artifact_binding(identity, shared)
        source_key = 'source_hash' if unicorn else 'implementation_hash'
        if identity[source_key] != native_source_digest(self.method, self.entry['source_root']):
            raise ValueError('Native transition encoder training source fingerprint mismatch')
        if unicorn:
            if cfg['method_variant'] != 'SS' or state['iteration'] - 1 != self.entry['final_iteration_zero_based'] or identity['training_seed'] != self.entry['training_seed']:
                raise ValueError('UNICORN SS final checkpoint identity invalid')
            model_state = state['model']
            if model_state['main_updates'] != a['num_iters'] * a['rl_updates_per_iter']:
                raise ValueError('UNICORN final update count differs')
            parts = model_state['modules']
            # Native SS samples Gaussian transition embeddings, then tanh and mean.
            self.report['terminal_input_semantics'] = 'native episode-end channel, not an absorbing-terminal label'
        else:
            if cfg['method_variant'] != 'generative' or cfg['seed'] != self.entry['training_seed'] or cfg['smoke']:
                raise ValueError('Expected the full generative CORRO main run')
            progress = state['progress']
            if progress['stage'] != 'complete' or progress['policy_updates'] != cfg['num_iters'] * cfg['rl_updates_per_iter']:
                raise ValueError('CORRO native training incomplete')
            parts = state['modules']
            context_cls = importlib.import_module('models.encoder').SelfAttnEncoder
            context = context_cls(input_dim=self.latent_dim, num_output_mlp=0)
            context.load_state_dict(parts['context'], strict=True)
            self.modules['context'] = context
        encoder.load_state_dict(parts['encoder'], strict=True)
        policy.load_state_dict(policy_state(parts['agent']), strict=True)
        self.modules.update(encoder=encoder, policy=policy)

    def prepared(self, raw):
        validate_context(raw, self.obs_dim, self.action_dim)
        out = {k: self.tensor(v) for k, v in raw.items()}
        for key in ('observations', 'next_observations'):
            out[key] = self.tensor(normalize(raw[key], self.stats))
        return out

    @torch.no_grad()
    def infer(self, raw, representation=False):
        batch = self.prepared(raw)
        if self.agent is not None:
            self.agent.clear_z()
            context = torch.cat([batch[k] for k in self.fields], dim=-1).unsqueeze(0)
            self.agent.infer_posterior(context)
            return (self.agent.z_means if representation else self.agent.z).detach().clone()
        encoder = self.modules['encoder']
        if self.method == 'CORRO':
            rows = encoder(*(batch[k] for k in self.fields))
            return self.modules['context'](rows.unsqueeze(0)).detach()
        if representation:
            mean = encoder.encoder(*(batch[k] for k in self.fields))
            variance = encoder.log_var.exp().clamp(0.1, 10.0).expand_as(mean)
            expected, error = tanh_gaussian_mean(mean, variance)
            self.report['quadrature_128_256_max_error'] = error
            return expected.mean(0, keepdim=True).detach()
        result, _ = encoder.context_encoding(obs=batch['observations'].unsqueeze(1),
                    actions=batch['actions'].unsqueeze(1), rewards=batch['rewards'].unsqueeze(1),
                    next_obs=batch['next_observations'].unsqueeze(1), terms=batch['episode_end'].unsqueeze(1))
        return result.detach()

    @torch.no_grad()
    def act(self, observation, condition):
        obs = normalize(np.asarray(observation), self.stats)
        if self.agent is not None:
            self.agent.z = condition
            action = self.agent.get_action(obs, deterministic=True)[0]
        else:
            inputs = torch.cat([self.tensor(obs).reshape(1, -1), condition], dim=-1)
            action = self.modules['policy'](inputs, deterministic=True)[0][0].cpu().numpy()
        action = np.asarray(action)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1 + 1e-6:
            raise ValueError('Invalid native normalized action')
        return np.clip(action, -1, 1)
