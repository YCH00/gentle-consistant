import os
import torch
import torch.optim as optim
import numpy as np
import copy
import time
import rlkit.torch.pytorch_util as ptu
from torch import nn as nn
import torch.nn.functional as F 
from collections import OrderedDict
from rlkit.core import logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import OfflineMetaRLAlgorithm
from rlkit.data_management.env_replay_buffer import MultiTaskContextBuffer

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
        self.virtual_interpolation_lambda_max = kwargs.get('virtual_interpolation_lambda_max', 0.2)
        self.virtual_interpolation_max_distance = kwargs.get('virtual_interpolation_max_distance', None)
        self.virtual_task_warmup_steps      = int(kwargs.get('virtual_task_warmup_steps', 200))
        self.virtual_cycle_filter_pool_factor = int(kwargs.get('virtual_cycle_filter_pool_factor', 4))
        self.virtual_cycle_filter_threshold = kwargs.get('virtual_cycle_filter_threshold', None)
        self.consistency_loss_weight        = kwargs.get('consistency_loss_weight', 1.0)
        self.virtual_policy_weight          = kwargs.get('virtual_policy_weight', 0.0)
        self.consistency_use_policy_relabel_data = kwargs.get('consistency_use_policy_relabel_data', True)
        self.virtual_policy_use_policy_relabel_data = kwargs.get('virtual_policy_use_policy_relabel_data', True)
        self.virtual_policy_adaptive_lambda = kwargs.get('virtual_policy_adaptive_lambda', True)
        self.virtual_policy_q_clip          = kwargs.get('virtual_policy_q_clip', None)
        self.virtual_policy_warmup_steps    = int(kwargs.get('virtual_policy_warmup_steps', 0))

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
    def _score_virtual_task_embeddings(self, virtual_task_z, batch_size):
        if virtual_task_z is None or len(virtual_task_z) == 0:
            return None

        anchor_task_indices = np.random.choice(
            self.train_tasks,
            size=virtual_task_z.size(0),
            replace=True,
        )
        fake_context = self._build_consistency_context(
            virtual_task_z,
            batch_size,
            anchor_task_indices=anchor_task_indices,
        )
        if fake_context is None:
            return None

        reencoded_z = self._get_context_embedding(fake_context, sample=False)
        return torch.mean((reencoded_z - virtual_task_z) ** 2, dim=-1)

    @torch.no_grad()
    def _sample_virtual_task_embeddings(self, batch_size):
        self.loss['virtual_warmup_active'] = 0
        self.loss['virtual_candidate_pool_size'] = 0
        self.loss['virtual_cycle_error_mean'] = 0.0
        self.loss['virtual_cycle_error_max'] = 0.0

        if self.n_vt <= 0 or len(self.train_tasks) <= 1:
            return None
        if self._n_train_steps_total < self.virtual_task_warmup_steps:
            self.loss['virtual_warmup_active'] = 1
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
        pool_factor = max(1, int(self.virtual_cycle_filter_pool_factor))
        target_pool_size = max(self.n_vt, self.n_vt * pool_factor)
        max_attempts = max(target_pool_size * 4, target_pool_size)
        attempts = 0
        while len(virtual_zs) < target_pool_size and attempts < max_attempts:
            attempts += 1
            anchor_idx = np.random.randint(num_real_tasks)
            neighbor_choices = nearest_neighbors[anchor_idx]
            neighbor_idx = neighbor_choices[np.random.randint(neighbor_k)].item()
            interpolation = torch.rand(1, 1, device=real_task_z.device) * lambda_max
            candidate_z = (
                (1.0 - interpolation) * real_task_z[anchor_idx:anchor_idx + 1]
                + interpolation * real_task_z[neighbor_idx:neighbor_idx + 1]
            )
            if max_distance is not None:
                nearest_dist = torch.norm(real_task_z - candidate_z, dim=-1).min()
                if nearest_dist.item() > max_distance:
                    continue
            virtual_zs.append(candidate_z)

        if len(virtual_zs) == 0:
            return None

        candidate_pool = torch.cat(virtual_zs, dim=0)
        self.loss['virtual_candidate_pool_size'] = candidate_pool.size(0)
        cycle_threshold = self.virtual_cycle_filter_threshold
        if cycle_threshold is not None:
            cycle_threshold = float(cycle_threshold)

        if pool_factor > 1 or cycle_threshold is not None:
            cycle_errors = self._score_virtual_task_embeddings(candidate_pool, batch_size)
            if cycle_errors is None:
                return None
            if cycle_threshold is not None:
                keep_mask = cycle_errors <= cycle_threshold
                if not torch.any(keep_mask):
                    self.loss['virtual_cycle_error_mean'] = cycle_errors.mean().item()
                    self.loss['virtual_cycle_error_max'] = cycle_errors.max().item()
                    return None
                candidate_pool = candidate_pool[keep_mask]
                cycle_errors = cycle_errors[keep_mask]

            order = torch.argsort(cycle_errors)
            selected_count = min(self.n_vt, candidate_pool.size(0))
            selected_order = order[:selected_count]
            selected_z = candidate_pool[selected_order]
            selected_errors = cycle_errors[selected_order]
            self.loss['virtual_cycle_error_mean'] = selected_errors.mean().item()
            self.loss['virtual_cycle_error_max'] = selected_errors.max().item()
            return selected_z

        return candidate_pool[:self.n_vt]

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

    def _compute_virtual_policy_loss(self, virtual_task_z, batch_size, use_policy_relabel_data=None):
        if virtual_task_z is None:
            zero = ptu.zeros(1).squeeze()
            return zero, zero, zero, zero

        virtual_task_z = virtual_task_z.detach()
        num_virtual_tasks = len(virtual_task_z)
        if num_virtual_tasks == 0:
            zero = ptu.zeros(1).squeeze()
            return zero, zero, zero, zero

        anchor_task_indices = np.random.choice(self.train_tasks, size=num_virtual_tasks, replace=True)
        anchor_context = self.sample_context(anchor_task_indices, b_size=batch_size)
        anchor_obs = anchor_context[:, :, :self.obs_dim]
        # anchor_actions = anchor_context[:, :, self.obs_dim:self.obs_dim + self.action_dim]
        repeated_virtual_z = virtual_task_z.unsqueeze(1).expand(-1, batch_size, -1)
        if use_policy_relabel_data is None:
            use_policy_relabel_data = self.virtual_policy_use_policy_relabel_data

        if use_policy_relabel_data:
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
            )[0]
        else:
            fake_task_indices = self._sample_mismatched_task_indices(anchor_task_indices)
            fake_context = self.sample_context(fake_task_indices, b_size=batch_size)
            fake_actions = fake_context[:, :, self.obs_dim:self.obs_dim + self.action_dim]
            virtual_actions = fake_actions.reshape(-1, self.action_dim)
        virtual_q = self._min_q(
            num_virtual_tasks,
            batch_size,
            anchor_obs.reshape(-1, self.obs_dim),
            virtual_actions,
            repeated_virtual_z.reshape(-1, self.latent_dim),
        )
        if self.virtual_policy_q_clip is not None:
            virtual_q = torch.clamp(virtual_q, -self.virtual_policy_q_clip, self.virtual_policy_q_clip)

        virtual_q_mean = virtual_q.mean()
        virtual_q_abs_mean = virtual_q.abs().mean()
        if self.virtual_policy_adaptive_lambda:
            virtual_policy_lambda = self.bc_weight / virtual_q_abs_mean.detach().clamp(min=1e-6)
        else:
            virtual_policy_lambda = torch.ones(1, device=virtual_q.device).squeeze()
        virtual_policy_loss = -virtual_policy_lambda * virtual_q_mean
        return virtual_policy_loss, virtual_q_mean, virtual_q_abs_mean, virtual_policy_lambda
    
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
        with torch.no_grad():
            virtual_task_z = self._sample_virtual_task_embeddings(c_b)
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
        encoder_total_loss = context_loss + self.consistency_loss_weight * consistency_loss
        
        self.context_optimizer.zero_grad()
        encoder_total_loss.backward()
        self.context_optimizer.step()
        
        q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
        q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
        with torch.no_grad():
            target_q1 = self.target_qf1(t, b, next_obs, next_actions, task_z)
            target_q2 = self.target_qf2(t, b, next_obs, next_actions, task_z)
            target_q = torch.min(target_q1, target_q2)
            rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
            # scale rewards for Bellman update
            rewards_flat = rewards_flat * self.reward_scale
            terms_flat = terms.view(self.batch_size * num_tasks, -1)
            target_q = rewards_flat + (1. - terms_flat) * self.discount * target_q
        qf_loss = torch.mean((q1_pred - target_q) ** 2) + torch.mean((q2_pred - target_q) ** 2)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(target_q).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self._set_requires_grad(self.qf1, False)
        self._set_requires_grad(self.qf2, False)
        Q = self._min_q(t, b, obs, new_actions, task_z)
        lmbda = self.bc_weight/Q.abs().mean().detach()
        policy_loss = -lmbda * Q.mean()
        bc_loss = F.mse_loss(new_actions, actions)
        virtual_policy_loss = ptu.zeros(1).squeeze()
        virtual_q_mean = ptu.zeros(1).squeeze()
        virtual_q_abs_mean = ptu.zeros(1).squeeze()
        virtual_policy_lambda = ptu.zeros(1).squeeze()
        if self.virtual_policy_weight > 0 and virtual_task_z is not None and self._n_train_steps_total >= self.virtual_policy_warmup_steps:
            virtual_policy_loss, virtual_q_mean, virtual_q_abs_mean, virtual_policy_lambda = self._compute_virtual_policy_loss(virtual_task_z, c_b)

        policy_total_loss = policy_loss + bc_loss + self.virtual_policy_weight * virtual_policy_loss
        self.loss["policy_loss"] = policy_loss.item()
        self.loss["bc_loss"] = bc_loss.item()
        self.loss["virtual_policy_loss"] = virtual_policy_loss.item()
        self.loss["virtual_q_mean"] = virtual_q_mean.item()
        self.loss["virtual_q_abs_mean"] = virtual_q_abs_mean.item()
        self.loss["virtual_policy_lambda"] = virtual_policy_lambda.item()
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
            self.eval_statistics['Virtual Policy Loss'] = np.mean(ptu.get_numpy(
                virtual_policy_loss
            ))
            self.eval_statistics['Virtual Q Mean'] = np.mean(ptu.get_numpy(
                virtual_q_mean
            ))
            self.eval_statistics['Virtual Q Abs Mean'] = np.mean(ptu.get_numpy(
                virtual_q_abs_mean
            ))
            self.eval_statistics['Virtual Policy Lambda'] = np.mean(ptu.get_numpy(
                virtual_policy_lambda
            ))
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q Predictions',  ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu',      ptu.get_numpy(policy_mean)))
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)
    
