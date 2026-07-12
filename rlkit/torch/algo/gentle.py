import os
import torch
import torch.optim as optim
import numpy as np
import copy
import time
import rlkit.torch.pytorch_util as ptu
from torch import nn as nn
import torch.nn.functional as F 
from torch.distributions import Dirichlet
from collections import OrderedDict
from rlkit.core import logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import OfflineMetaRLAlgorithm
from rlkit.data_management.env_replay_buffer import MultiTaskContextBuffer

class VirtualTaskReplayBuffer(object):
    def __init__(self, max_replay_buffer_size, obs_dim, action_dim, latent_dim):
        self._max_replay_buffer_size = int(max_replay_buffer_size)
        self._observations = np.zeros((self._max_replay_buffer_size, obs_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_replay_buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros((self._max_replay_buffer_size, 1), dtype=np.float32)
        self._next_obs = np.zeros((self._max_replay_buffer_size, obs_dim), dtype=np.float32)
        self._terminals = np.zeros((self._max_replay_buffer_size, 1), dtype='uint8')
        self._task_z = np.zeros((self._max_replay_buffer_size, latent_dim), dtype=np.float32)
        self.clear()

    def clear(self):
        self._top = 0
        self._size = 0

    def size(self):
        return self._size

    def num_steps_can_sample(self):
        return self._size

    def add_batch(self, observations, actions, rewards, next_observations, terminals, task_z):
        n_samples = observations.shape[0]
        if n_samples <= 0:
            return
        if n_samples > self._max_replay_buffer_size:
            observations = observations[-self._max_replay_buffer_size:]
            actions = actions[-self._max_replay_buffer_size:]
            rewards = rewards[-self._max_replay_buffer_size:]
            next_observations = next_observations[-self._max_replay_buffer_size:]
            terminals = terminals[-self._max_replay_buffer_size:]
            task_z = task_z[-self._max_replay_buffer_size:]
            n_samples = self._max_replay_buffer_size

        indices = (np.arange(n_samples) + self._top) % self._max_replay_buffer_size
        self._observations[indices] = observations
        self._actions[indices] = actions
        self._rewards[indices] = rewards
        self._next_obs[indices] = next_observations
        self._terminals[indices] = terminals
        self._task_z[indices] = task_z
        self._top = (self._top + n_samples) % self._max_replay_buffer_size
        self._size = min(self._size + n_samples, self._max_replay_buffer_size)

    def random_batch(self, batch_size):
        assert self._size > 0
        indices = np.random.randint(0, self._size, batch_size)
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_observations=self._next_obs[indices],
            terminals=self._terminals[indices],
            task_z=self._task_z[indices],
        )

class GENTLE(OfflineMetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,
            goal_radius=1,
            obs_normalizer=None,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            goal_radius=goal_radius,
            obs_normalizer=obs_normalizer,
            **kwargs
        )

        self.latent_dim                     = latent_dim
        self.soft_target_tau                = kwargs['soft_target_tau']
        self.sparse_rewards                 = kwargs['sparse_rewards']
        self.use_next_obs_in_context        = kwargs['use_next_obs_in_context']
        self.policy_lr                      = kwargs['policy_lr']
        self.qf_lr                          = kwargs['qf_lr']
        self.context_lr                     = kwargs['context_lr']

        self.policy_noise                   = kwargs['policy_noise']
        self.noise_clip                     = kwargs['noise_clip']
        self.policy_freq                    = kwargs['policy_freq']
        self.bc_weight                      = kwargs['bc_weight']
        self.recon_loss_weight              = kwargs['recon_loss_weight']
        self.relabel_data_ratio             = kwargs['relabel_data_ratio']
        self.relabel_buffer_size            = kwargs['relabel_buffer_size']
        self.num_aug_neg_tasks              = kwargs['num_aug_neg_tasks']
        self.max_action                     = 1.
        self.n_vt                           = kwargs.get('n_vt', 0)
        self.M                              = kwargs.get('M', 2)
        self.beta                           = kwargs.get('beta', 1.0)
        self.virtual_task_generation_mode   = kwargs.get('virtual_task_generation_mode', 'local').lower()
        if self.virtual_task_generation_mode not in ('local', 'global', 'gaussian'):
            raise ValueError(
                "virtual_task_generation_mode must be one of 'local', 'global', or 'gaussian', "
                "got '{}'".format(self.virtual_task_generation_mode)
            )
        self.virtual_interpolation_lambda_max = kwargs.get('virtual_interpolation_lambda_max', 0.2)
        self.virtual_interpolation_max_distance = kwargs.get('virtual_interpolation_max_distance', None)
        self.virtual_gaussian_noise_std = float(kwargs.get('virtual_gaussian_noise_std', 0.05))
        if self.virtual_gaussian_noise_std < 0.0:
            raise ValueError(
                "virtual_gaussian_noise_std must be non-negative, "
                "got {}".format(self.virtual_gaussian_noise_std)
            )
        self.consistency_loss_weight        = kwargs.get('consistency_loss_weight', 1.0)
        self.consistency_use_policy_relabel_data = kwargs.get('consistency_use_policy_relabel_data', True)
        self.virtual_transition_buffer_size = int(kwargs.get('virtual_transition_buffer_size', 50000))
        self.virtual_transition_batch_size  = int(kwargs.get('virtual_transition_batch_size', 256))
        self.virtual_transition_loss_weight = float(kwargs.get('virtual_transition_loss_weight', 1.0))
        self.virtual_transition_weight_schedule = kwargs.get(
            'virtual_transition_weight_schedule',
            'constant',
        ).lower()
        if self.virtual_transition_weight_schedule not in ('constant', 'linear_decay'):
            raise ValueError(
                "virtual_transition_weight_schedule must be either 'constant' "
                "or 'linear_decay', got '{}'".format(self.virtual_transition_weight_schedule)
            )
        self.virtual_transition_weight_decay_start_itr = int(
            kwargs.get('virtual_transition_weight_decay_start_itr', 0)
        )
        self.virtual_transition_weight_decay_end_itr = kwargs.get(
            'virtual_transition_weight_decay_end_itr',
            None,
        )
        if self.virtual_transition_weight_decay_end_itr is None:
            self.virtual_transition_weight_decay_end_itr = max(
                self.virtual_transition_weight_decay_start_itr + 1,
                self.num_iterations - 1,
            )
        self.virtual_transition_weight_decay_end_itr = int(
            self.virtual_transition_weight_decay_end_itr
        )
        self.virtual_transition_final_loss_weight = float(
            kwargs.get('virtual_transition_final_loss_weight', 0.0)
        )
        if self.virtual_transition_loss_weight < 0.0:
            raise ValueError('virtual_transition_loss_weight must be non-negative')
        if self.virtual_transition_final_loss_weight < 0.0:
            raise ValueError('virtual_transition_final_loss_weight must be non-negative')
        if self.virtual_transition_weight_decay_start_itr < 0:
            raise ValueError('virtual_transition_weight_decay_start_itr must be non-negative')
        if (
            self.virtual_transition_weight_schedule == 'linear_decay'
            and self.virtual_transition_weight_decay_end_itr <= self.virtual_transition_weight_decay_start_itr
        ):
            raise ValueError(
                'virtual_transition_weight_decay_end_itr must be larger than '
                'virtual_transition_weight_decay_start_itr when using linear_decay'
            )
        self.virtual_transition_use_policy_actions = kwargs.get('virtual_transition_use_policy_actions', False)
        if self.virtual_transition_use_policy_actions and not self.use_next_obs_in_context:
            print(
                'Warning: virtual_transition_use_policy_actions=True with '
                'use_next_obs_in_context=False reuses real next_obs, so generated virtual '
                'transitions may pair policy actions with unmatched next observations.'
            )

        self.loss                           = {}
        self.plotter                        = plotter
        self.render_eval_paths              = render_eval_paths

        self.qf1, self.qf2 = nets[1:3]
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.context_decoder = nets[3]
        self.task_dynamics = nets[4]
        self.policy_optimizer               = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.qf1_optimizer                  = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer                  = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        self.context_optimizer              = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)

        self._set_requires_grad(self.agent.context_encoder, True)
        self._set_requires_grad(self.context_decoder, False)
        

        self._num_steps                     = 0
        self._visit_num_steps_train         = 10

        self.relabel_buffer     = MultiTaskContextBuffer(self.relabel_buffer_size, env, self.train_tasks, kwargs['context_dim'])
        self.virtual_transition_buffer = None
        if self.virtual_transition_buffer_size > 0 and self.virtual_transition_batch_size > 0:
            self.virtual_transition_buffer = VirtualTaskReplayBuffer(
                self.virtual_transition_buffer_size,
                self.obs_dim,
                self.action_dim,
                self.latent_dim,
            )

    ###### Torch stuff #####
    @property
    def networks(self):
        nets = self.agent.networks + [self.agent] + [self.context_policy, self.context_policy.policy] + [self.qf1, self.qf2, self.target_qf1, self.target_qf2, self.context_decoder]
        return nets

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.task_dynamics is not None:
            self.task_dynamics.to(device)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        #print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    def _set_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _get_virtual_transition_loss_weight(self):
        if self.virtual_transition_weight_schedule == 'constant':
            return self.virtual_transition_loss_weight

        current_itr = int(getattr(self, 'itr', 0))
        start_itr = self.virtual_transition_weight_decay_start_itr
        end_itr = self.virtual_transition_weight_decay_end_itr
        if current_itr <= start_itr:
            return self.virtual_transition_loss_weight
        if current_itr >= end_itr:
            return self.virtual_transition_final_loss_weight

        progress = float(current_itr - start_itr) / float(end_itr - start_itr)
        return (
            self.virtual_transition_loss_weight
            + progress * (
                self.virtual_transition_final_loss_weight
                - self.virtual_transition_loss_weight
            )
        )

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]
    
    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.train_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices, b_size=None):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size if b_size is None else b_size)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context] # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        # self.meta_batch * self.embedding_batch_size * sum_dim(o, a, r, no, t)
        return context

    def sample_transition_batch(self, indices, b_size):
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.train_buffer.random_batch(idx, batch_size=b_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_virtual_sac(self, batch_size):
        if self.virtual_transition_buffer is None:
            return None
        if self.virtual_transition_buffer.num_steps_can_sample() <= 0:
            return None
        batch_size = min(batch_size, self.virtual_transition_buffer.num_steps_can_sample())
        batch = ptu.np_to_pytorch_batch(self.virtual_transition_buffer.random_batch(batch_size))
        obs = batch['observations'][None, ...]
        actions = batch['actions'][None, ...]
        rewards = batch['rewards'][None, ...]
        next_obs = batch['next_observations'][None, ...]
        terms = batch['terminals'][None, ...]
        task_z = batch['task_z']
        return obs, actions, rewards, next_obs, terms, task_z

    @torch.no_grad()
    def add_virtual_transitions_to_buffer(self, virtual_task_z, batch_size):
        if self.virtual_transition_buffer is None or virtual_task_z is None:
            return 0
        if len(virtual_task_z) == 0:
            return 0

        num_virtual_tasks = virtual_task_z.size(0)
        anchor_task_indices = np.random.choice(self.train_tasks, size=num_virtual_tasks, replace=True)
        anchor_obs, anchor_actions, _, anchor_next_obs, anchor_terms = self.sample_transition_batch(
            anchor_task_indices,
            batch_size,
        )
        repeated_virtual_z = virtual_task_z.detach().unsqueeze(1).expand(-1, batch_size, -1)

        if self.virtual_transition_use_policy_actions:
            policy_inputs = torch.cat(
                [
                    anchor_obs.reshape(-1, self.obs_dim),
                    repeated_virtual_z.reshape(-1, self.latent_dim),
                ],
                dim=-1,
            )
            virtual_actions = self.agent.policy(
                num_virtual_tasks,
                batch_size,
                policy_inputs,
                reparameterize=True,
                return_log_prob=True,
            )[0].reshape(num_virtual_tasks, batch_size, self.action_dim)
        else:
            virtual_actions = anchor_actions

        decoder_output = self.context_decoder(anchor_obs, virtual_actions, repeated_virtual_z)
        if self.use_next_obs_in_context:
            virtual_rewards = decoder_output[:, :, :1]
            virtual_next_obs = decoder_output[:, :, 1:]
        else:
            virtual_rewards = decoder_output
            virtual_next_obs = anchor_next_obs

        flat_obs = anchor_obs.reshape(-1, self.obs_dim)
        flat_actions = virtual_actions.reshape(-1, self.action_dim)
        flat_rewards = virtual_rewards.reshape(-1, 1)
        flat_next_obs = virtual_next_obs.reshape(-1, self.obs_dim)
        flat_terms = anchor_terms.reshape(-1, 1)
        flat_task_z = repeated_virtual_z.reshape(-1, self.latent_dim)

        self.virtual_transition_buffer.add_batch(
            ptu.get_numpy(flat_obs),
            ptu.get_numpy(flat_actions),
            ptu.get_numpy(flat_rewards),
            ptu.get_numpy(flat_next_obs),
            ptu.get_numpy(flat_terms),
            ptu.get_numpy(flat_task_z),
        )
        return flat_obs.size(0)

    def _product_of_gaussians(self, mus, sigmas_squared):
        sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
        sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
        mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
        return mu, sigma_squared

    def _get_context_embedding(self, context, sample=False):
        params = self.agent.context_encoder(context)
        params = params.view(context.size(0), -1, self.agent.context_encoder.output_size)

        if self.agent.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [self._product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            z_means = torch.stack([p[0] for p in z_params])
            z_vars = torch.stack([p[1] for p in z_params])
            if sample:
                posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
                z = [d.rsample() for d in posteriors]
                return torch.stack(z)
            return z_means

        return torch.mean(params, dim=1)

    @torch.no_grad()
    def _build_consistency_context(self, task_z, batch_size, anchor_task_indices=None, use_policy_relabel_data=None):
        if task_z is None or len(task_z) == 0:
            return None

        task_z = task_z.detach()
        num_tasks = task_z.size(0)
        if anchor_task_indices is None:
            anchor_task_indices = np.random.choice(self.train_tasks, size=num_tasks, replace=True)
        else:
            anchor_task_indices = np.asarray(anchor_task_indices)
            if len(anchor_task_indices) != num_tasks:
                raise ValueError(
                    'anchor_task_indices must have the same length as task_z: '
                    '{} vs {}'.format(len(anchor_task_indices), num_tasks)
                )

        anchor_context = self.sample_context(anchor_task_indices, b_size=batch_size)
        anchor_obs = anchor_context[:, :, :self.obs_dim]
        repeated_task_z = task_z.unsqueeze(1).expand(-1, batch_size, -1)
        if use_policy_relabel_data is None:
            use_policy_relabel_data = self.consistency_use_policy_relabel_data

        if use_policy_relabel_data:
            policy_inputs = torch.cat(
                [
                    anchor_obs.reshape(-1, self.obs_dim),
                    repeated_task_z.reshape(-1, self.latent_dim),
                ],
                dim=-1,
            )
            fake_actions = self.agent.policy(
                num_tasks,
                batch_size,
                policy_inputs,
                reparameterize=True,
                return_log_prob=True,
            )[0].reshape(num_tasks, batch_size, self.action_dim)
        else:
            fake_task_indices = self._sample_mismatched_task_indices(anchor_task_indices)
            fake_context = self.sample_context(fake_task_indices, b_size=batch_size)
            fake_actions = fake_context[:, :, self.obs_dim:self.obs_dim + self.action_dim]

        fake_r_next_s = self.context_decoder(anchor_obs, fake_actions, repeated_task_z)
        return torch.cat([anchor_obs, fake_actions, fake_r_next_s], dim=-1)

    @torch.no_grad()
    def _sample_virtual_task_embeddings(self, batch_size):
        if self.virtual_task_generation_mode == 'global':
            return self._sample_global_virtual_task_embeddings(batch_size)
        if self.virtual_task_generation_mode == 'gaussian':
            return self._sample_gaussian_virtual_task_embeddings(batch_size)
        return self._sample_local_virtual_task_embeddings(batch_size)

    @torch.no_grad()
    def _sample_global_virtual_task_embeddings(self, batch_size):
        if self.n_vt <= 0 or len(self.train_tasks) == 0:
            return None

        mixing_num_tasks = max(1, int(self.M))
        beta = float(self.beta)
        virtual_zs = []
        for _ in range(self.n_vt):
            mixing_task_indices = np.random.choice(
                self.train_tasks,
                mixing_num_tasks,
                replace=True,
            )
            mixing_context = self.sample_context(mixing_task_indices, b_size=batch_size)
            mixing_task_z = self._get_context_embedding(mixing_context, sample=False)
            alpha = Dirichlet(
                torch.ones(mixing_num_tasks, device=mixing_task_z.device)
            ).sample().unsqueeze(0)
            alpha = alpha * beta - (beta - 1.0) / mixing_num_tasks
            virtual_zs.append(alpha @ mixing_task_z)

        if len(virtual_zs) == 0:
            return None
        return torch.cat(virtual_zs, dim=0)

    @torch.no_grad()
    def _sample_local_virtual_task_embeddings(self, batch_size):
        if self.n_vt <= 0 or len(self.train_tasks) <= 1:
            return None

        real_task_indices = np.asarray(self.train_tasks)
        real_context = self.sample_context(real_task_indices, b_size=batch_size)
        real_task_z = self._get_context_embedding(real_context, sample=False)
        num_real_tasks = real_task_z.size(0)
        if num_real_tasks <= 1:
            return None

        neighbor_k = min(max(1, int(self.M)), num_real_tasks - 1)
        pairwise_dist = torch.cdist(real_task_z, real_task_z)
        pairwise_dist.fill_diagonal_(float('inf'))
        nearest_neighbors = torch.topk(
            pairwise_dist,
            k=neighbor_k,
            dim=1,
            largest=False,
        ).indices

        lambda_max = max(0.0, float(self.virtual_interpolation_lambda_max))
        max_distance = self.virtual_interpolation_max_distance
        if max_distance is not None:
            max_distance = float(max_distance)

        virtual_zs = []
        max_attempts = max(self.n_vt * 4, self.n_vt)
        attempts = 0
        while len(virtual_zs) < self.n_vt and attempts < max_attempts:
            attempts += 1
            anchor_idx = np.random.randint(num_real_tasks)
            neighbor_choices = nearest_neighbors[anchor_idx]
            neighbor_z = real_task_z[neighbor_choices]
            neighbor_weights = torch.rand(1, neighbor_k, device=real_task_z.device)
            neighbor_weights = neighbor_weights / neighbor_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            mixed_neighbor_z = neighbor_weights @ neighbor_z
            interpolation = torch.rand(1, 1, device=real_task_z.device) * lambda_max
            candidate_z = (
                (1.0 - interpolation) * real_task_z[anchor_idx:anchor_idx + 1]
                + interpolation * mixed_neighbor_z
            )
            if max_distance is not None:
                nearest_dist = torch.norm(real_task_z - candidate_z, dim=-1).min()
                if nearest_dist.item() > max_distance:
                    continue
            virtual_zs.append(candidate_z)

        if len(virtual_zs) == 0:
            return None

        return torch.cat(virtual_zs, dim=0)

    @torch.no_grad()
    def _sample_gaussian_virtual_task_embeddings(self, batch_size):
        if self.n_vt <= 0 or len(self.train_tasks) == 0:
            return None

        base_task_indices = np.random.choice(
            self.train_tasks,
            size=self.n_vt,
            replace=True,
        )
        base_context = self.sample_context(base_task_indices, b_size=batch_size)
        base_task_z = self._get_context_embedding(base_context, sample=False)
        if self.virtual_gaussian_noise_std == 0.0:
            return base_task_z

        noise = torch.randn_like(base_task_z) * self.virtual_gaussian_noise_std
        return base_task_z + noise

    @torch.no_grad()
    def get_virtual_task_embeddings_for_vis(self, n_points):
        if self.n_vt <= 0:
            return None

        virtual_zs = []
        for _ in range(n_points):
            virtual_task_z = self._sample_virtual_task_embeddings(self.online_sample_num)
            if virtual_task_z is None:
                return None
            virtual_zs.append(ptu.get_numpy(virtual_task_z))

        if len(virtual_zs) == 0:
            return None

        virtual_zs = np.concatenate(virtual_zs, axis=0)
        return virtual_zs[np.newaxis, ...]

    def _compute_consistency_loss(self, task_z, batch_size, anchor_task_indices=None, use_policy_relabel_data=None):
        if task_z is None:
            return ptu.zeros(1).squeeze()

        task_z = task_z.detach()
        fake_context = self._build_consistency_context(
            task_z,
            batch_size,
            anchor_task_indices=anchor_task_indices,
            use_policy_relabel_data=use_policy_relabel_data,
        )
        if fake_context is None:
            return ptu.zeros(1).squeeze()
        fake_task_z = self._get_context_embedding(fake_context, sample=False)
        return F.mse_loss(fake_task_z, task_z)
    
    def _sample_mismatched_task_indices(self, task_indices):
        task_indices = np.asarray(task_indices)
        train_tasks = np.asarray(self.train_tasks)
        if len(train_tasks) <= 1:
            return np.random.choice(train_tasks, size=len(task_indices), replace=True)
        return np.asarray([
            np.random.choice(train_tasks[train_tasks != task_idx])
            for task_idx in task_indices
        ])    
    
    def get_relabel_output(self, obs, actions, task_indices):
        with torch.no_grad():
            relabel_output, relabel_std = self.task_dynamics.step(obs, actions, task_indices, return_std=True)
        return relabel_output, relabel_std

    def make_relabel(self, _n=10):
        sample_b_s = self.embedding_batch_size * _n
        indices = np.array(self.train_tasks)
        
        context_batch = np.concatenate([self.unpack_context_batch(self.train_buffer.random_batch(idx, sample_b_s)) for idx in indices])
        context_batch = ptu.from_numpy(context_batch)
        c_mb, c_b, _ = context_batch.shape

        relabel_context = copy.deepcopy(context_batch)
        relabel_obs = relabel_context[:,:,:self.obs_dim].view(c_mb * c_b, -1)
        
        with torch.no_grad():
            relabel_actions = self.agent.get_target_policy_action(relabel_context[:,:,:self.obs_dim], context_batch, task_indices=indices, reinfer=True)

        relabel_context[:,:,self.obs_dim:self.obs_dim+self.action_dim] = relabel_actions.reshape(c_mb, c_b, -1)
        relabel_output, relabel_std = self.get_relabel_output(relabel_obs, relabel_actions, task_indices=indices)
        
        relabel_context[:,:,self.obs_dim+self.action_dim:] = relabel_output.view(c_mb, c_b, -1)
        sorted_ind = torch.argsort(relabel_std, dim=-1)
        sorted_relabel = torch.cat([relabel_context[i, sorted_ind[i], :] for i in range(c_mb)]).reshape(c_mb, c_b, -1)

        self.relabel_buffer.add_sample(indices, ptu.get_numpy(sorted_relabel))

        num_aug = self.num_aug_neg_tasks
        all_neg_indices = np.array([np.random.choice(list(indices)[0:i] + list(indices)[i+1:], num_aug, replace=False) for i in range(len(indices))])
        for i in range(num_aug):
            neg_indices = all_neg_indices[:, i]
            
            relabel_output, relabel_std = self.get_relabel_output(relabel_obs, relabel_actions, task_indices=neg_indices)
            relabel_context[:,:,self.obs_dim+self.action_dim:] = relabel_output.view(c_mb, c_b, -1)
            sorted_ind = torch.argsort(relabel_std, dim=-1)
            sorted_relabel = torch.cat([relabel_context[i, sorted_ind[i], :] for i in range(c_mb)]).reshape(c_mb, c_b, -1)

            self.relabel_buffer.add_sample(neg_indices, ptu.get_numpy(sorted_relabel))


    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size # NOTE: not meta batch!
        num_updates = self.embedding_batch_size // mb_size

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        if self._n_train_steps_total % self.num_train_steps_per_itr == 0:
            self.relabel_buffer.clear(self.train_tasks)
            self.make_relabel()

        # sample context batch
        relabel_data_size = int(self.embedding_batch_size * self.relabel_data_ratio)
        context_batch = self.sample_context(indices, b_size=self.embedding_batch_size-relabel_data_size)
        relabel_context = ptu.from_numpy(self.relabel_buffer.random_batch_task(relabel_data_size, indices))
        context_batch = torch.cat([context_batch, relabel_context], dim=1)

        z_means_lst = []
        z_vars_lst = []
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.loss['step'] = self._num_steps
            z_means, z_vars = self._take_step(indices, context)
            self._num_steps += 1
            z_means_lst.append(z_means[None, ...])
            z_vars_lst.append(z_vars[None, ...])
            # stop backprop
            self.agent.detach_z()
            torch.cuda.empty_cache()
        z_means = np.mean(np.concatenate(z_means_lst), axis=0)
        z_vars = np.mean(np.concatenate(z_vars_lst), axis=0)
        return z_means, z_vars

    def _min_q(self, t, b, obs, actions, task_z):
        q1 = self.qf1(t, b, obs, actions, task_z.detach())
        q2 = self.qf2(t, b, obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self, f, target_f):
        ptu.soft_update_from_to(f, target_f, self.soft_target_tau)

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            target_qf1=self.target_qf1.state_dict(),
            target_qf2=self.target_qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            target_policy=self.agent.target_policy.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            context_decoder=self.context_decoder.state_dict()
        )
        return snapshot

    def load_epoch_model(self, epoch, log_dir):
        path = log_dir
        try:
            self.agent.context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder_itr_{}.pth'.format(epoch))))
            self.agent.policy.load_state_dict(torch.load(os.path.join(path, 'policy_itr_{}.pth'.format(epoch))))
            self.agent.target_policy.load_state_dict(torch.load(os.path.join(path, 'target_policy_itr_{}.pth'.format(epoch))))
            self.qf1.load_state_dict(torch.load(os.path.join(path, 'qf1_itr_{}.pth'.format(epoch))))
            self.qf2.load_state_dict(torch.load(os.path.join(path, 'qf2_itr_{}.pth'.format(epoch))))
            self.target_qf1.load_state_dict(torch.load(os.path.join(path, 'target_qf1_itr_{}.pth'.format(epoch))))
            self.target_qf2.load_state_dict(torch.load(os.path.join(path, 'target_qf2_itr_{}.pth'.format(epoch))))
            return True
        except:
            print("epoch: {} is not ready".format(epoch))
            return False
    
    def _take_step(self, indices, context):
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(reward_in_context)

        num_tasks = len(indices)
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        policy_outputs, task_z, task_z_vars= self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        with torch.no_grad():
            next_actions = self.agent.get_target_policy_action(next_obs, context, task_indices=indices)	
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        next_actions = next_actions.view(t * b, -1)

        c_mb, c_b, _ = context.size()


        r_next_s = context[...,obs_dim+action_dim:]
        context_task_z = self.agent.z.unsqueeze(1).expand(-1, c_b, -1)
        pred_r_next_s = self.context_decoder(context[...,:obs_dim], context[...,obs_dim:obs_dim+action_dim], context_task_z)
        recon_loss = torch.mean((r_next_s - pred_r_next_s)**2)
        consistency_task_z = self.agent.z_means
        consistency_anchor_task_indices = np.asarray(indices)
        # consistency_anchor_task_indices = self._sample_mismatched_task_indices(indices)
        current_virtual_transition_loss_weight = self._get_virtual_transition_loss_weight()
        with torch.no_grad():
            virtual_task_z = self._sample_virtual_task_embeddings(c_b)
        if current_virtual_transition_loss_weight > 0.0:
            num_virtual_transitions_added = self.add_virtual_transitions_to_buffer(virtual_task_z, c_b)
        else:
            num_virtual_transitions_added = 0
        if virtual_task_z is not None:
            virtual_anchor_task_indices = np.random.choice(
                self.train_tasks,
                size=len(virtual_task_z),
                replace=True,
            )
            consistency_task_z = torch.cat([consistency_task_z, virtual_task_z], dim=0)
            consistency_anchor_task_indices = np.concatenate([
                consistency_anchor_task_indices,
                virtual_anchor_task_indices,
            ])
        consistency_loss = self._compute_consistency_loss(
            consistency_task_z,
            c_b,
            anchor_task_indices=consistency_anchor_task_indices,
        )
        context_loss = self.recon_loss_weight * recon_loss
        self.loss['recon_loss'] = recon_loss.item()
        self.loss['context_loss'] = context_loss.item()
        self.loss['consistency_loss'] = consistency_loss.item()
        self.loss['num_virtual_tasks'] = 0 if virtual_task_z is None else len(virtual_task_z)
        self.loss['num_virtual_transitions_added'] = num_virtual_transitions_added
        self.loss['virtual_transition_loss_weight_current'] = current_virtual_transition_loss_weight
        self.loss['virtual_transition_buffer_size'] = (
            0 if self.virtual_transition_buffer is None else self.virtual_transition_buffer.size()
        )
        encoder_total_loss = context_loss + self.consistency_loss_weight * consistency_loss
        
        self.context_optimizer.zero_grad()
        encoder_total_loss.backward()
        self.context_optimizer.step()
        
        real_sac_batch_size = t * b
        td3_obs = obs
        td3_actions = actions
        td3_rewards = rewards.view(real_sac_batch_size, -1)
        td3_next_obs = next_obs
        td3_terms = terms.view(real_sac_batch_size, -1)
        td3_task_z = task_z
        td3_next_actions = next_actions
        td3_new_actions = new_actions
        td3_weights = ptu.ones(real_sac_batch_size, 1)
        virtual_qf_loss = ptu.zeros(1).squeeze()
        virtual_bc_loss = ptu.zeros(1).squeeze()

        virtual_batch = None
        if current_virtual_transition_loss_weight > 0.0:
            virtual_batch = self.sample_virtual_sac(self.virtual_transition_batch_size)
        virtual_batch_size = 0
        if virtual_batch is not None:
            v_obs, v_actions, v_rewards, v_next_obs, v_terms, v_task_z = virtual_batch
            _, virtual_batch_size, _ = v_obs.size()
            flat_v_obs = v_obs.view(virtual_batch_size, -1)
            flat_v_actions = v_actions.view(virtual_batch_size, -1)
            flat_v_rewards = v_rewards.view(virtual_batch_size, -1)
            flat_v_next_obs = v_next_obs.view(virtual_batch_size, -1)
            flat_v_terms = v_terms.view(virtual_batch_size, -1)

            with torch.no_grad():
                virtual_next_actions = self.agent.get_target_policy_action(
                    v_next_obs,
                    None,
                    given_z=v_task_z,
                )
                virtual_noise = (
                    torch.randn_like(virtual_next_actions) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                virtual_next_actions = (
                    virtual_next_actions + virtual_noise
                ).clamp(-self.max_action, self.max_action)

            virtual_actor_inputs = torch.cat([flat_v_obs, v_task_z.detach()], dim=-1)
            virtual_new_actions = self.agent.policy(
                1,
                virtual_batch_size,
                virtual_actor_inputs,
                reparameterize=True,
                return_log_prob=True,
            )[0]

            td3_obs = torch.cat([td3_obs, flat_v_obs], dim=0)
            td3_actions = torch.cat([td3_actions, flat_v_actions], dim=0)
            td3_rewards = torch.cat([td3_rewards, flat_v_rewards], dim=0)
            td3_next_obs = torch.cat([td3_next_obs, flat_v_next_obs], dim=0)
            td3_terms = torch.cat([td3_terms, flat_v_terms], dim=0)
            td3_task_z = torch.cat([td3_task_z, v_task_z], dim=0)
            td3_next_actions = torch.cat([td3_next_actions, virtual_next_actions], dim=0)
            td3_new_actions = torch.cat([td3_new_actions, virtual_new_actions], dim=0)
            virtual_weights = ptu.ones(virtual_batch_size, 1) * current_virtual_transition_loss_weight
            td3_weights = torch.cat([td3_weights, virtual_weights], dim=0)

        td3_batch_size = td3_obs.size(0)
        td3_weight_sum = td3_weights.sum().clamp(min=1e-6)

        q1_pred = self.qf1(1, td3_batch_size, td3_obs, td3_actions, td3_task_z.detach())
        q2_pred = self.qf2(1, td3_batch_size, td3_obs, td3_actions, td3_task_z.detach())
        with torch.no_grad():
            target_q1 = self.target_qf1(
                1,
                td3_batch_size,
                td3_next_obs,
                td3_next_actions,
                td3_task_z,
            )
            target_q2 = self.target_qf2(
                1,
                td3_batch_size,
                td3_next_obs,
                td3_next_actions,
                td3_task_z,
            )
            target_q = torch.min(target_q1, target_q2)
            # scale rewards for Bellman update
            target_q = td3_rewards * self.reward_scale + (1. - td3_terms) * self.discount * target_q

        qf_element_loss = (q1_pred - target_q) ** 2 + (q2_pred - target_q) ** 2
        qf_loss = (qf_element_loss * td3_weights).sum() / td3_weight_sum
        real_qf_loss = torch.mean(qf_element_loss[:real_sac_batch_size])
        if virtual_batch_size > 0:
            virtual_qf_loss = torch.mean(qf_element_loss[real_sac_batch_size:])
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["real_qf_loss"] = real_qf_loss.item()
        self.loss["virtual_qf_loss"] = virtual_qf_loss.item()
        self.loss["virtual_transition_batch_size"] = virtual_batch_size
        self.loss["q_target"] = torch.mean(target_q).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self._set_requires_grad(self.qf1, False)
        self._set_requires_grad(self.qf2, False)
        Q = self._min_q(1, td3_batch_size, td3_obs, td3_new_actions, td3_task_z.detach())
        weighted_abs_q = (Q.abs() * td3_weights).sum() / td3_weight_sum
        lmbda = self.bc_weight / weighted_abs_q.detach().clamp(min=1e-6)
        policy_loss = -lmbda * (Q * td3_weights).sum() / td3_weight_sum
        bc_element_loss = torch.mean((td3_new_actions - td3_actions) ** 2, dim=1, keepdim=True)
        bc_loss = (bc_element_loss * td3_weights).sum() / td3_weight_sum
        if virtual_batch_size > 0:
            virtual_bc_loss = torch.mean(bc_element_loss[real_sac_batch_size:])

        policy_total_loss = policy_loss + bc_loss
        self.loss["policy_loss"] = policy_loss.item()
        self.loss["bc_loss"] = bc_loss.item()
        self.loss["virtual_transition_bc_mse"] = virtual_bc_loss.item()
        self.loss['encoder_total_loss'] = encoder_total_loss.item()
        self.loss['policy_total_loss'] = policy_total_loss.item()

        self.policy_optimizer.zero_grad()
        policy_total_loss.backward()
        self.policy_optimizer.step()
        self._set_requires_grad(self.qf1, True)
        self._set_requires_grad(self.qf2, True)

        self._update_target_network(self.qf1, self.target_qf1)
        self._update_target_network(self.qf2, self.target_qf2)
        self._update_target_network(self.agent.policy, self.agent.target_policy)

        # save some statistics for eval
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

            for i in range(len(self.agent.z_means[0])):
                z_mean = ptu.get_numpy(self.agent.z_means[0][i])
                name = 'Z mean train' + str(i)
                self.eval_statistics[name] = z_mean
            z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['task idx'] = indices[0]
            self.eval_statistics['Recon Loss'] = ptu.get_numpy(recon_loss)
            self.eval_statistics['Consistency Loss'] = ptu.get_numpy(consistency_loss)
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['Policy Total Loss'] = np.mean(ptu.get_numpy(
                policy_total_loss
            ))
            self.eval_statistics['BC Loss'] = np.mean(ptu.get_numpy(
                bc_loss
            ))
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['Virtual Transition QF Loss'] = np.mean(ptu.get_numpy(
                virtual_qf_loss
            ))
            self.eval_statistics['Virtual Transition BC MSE'] = np.mean(ptu.get_numpy(
                virtual_bc_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict('Q Predictions',  ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu',      ptu.get_numpy(policy_mean)))
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)
    
