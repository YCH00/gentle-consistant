"""The matched training step, kept separate from the historical GENTLE path."""
from collections import OrderedDict
import time
import numpy as np
import torch

from .losses import recovery_terms, critic_terms, actor_terms
from .randomness import isolated_rng


class MatchedStep:
    @staticmethod
    def _gradient_norm(loss, parameters):
        parameters = list(parameters)
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        return sum(gradient.detach().square().sum().item() for gradient in gradients if gradient is not None) ** 0.5

    def auxiliary_rng(self, purpose):
        return isolated_rng("component-ablations-v1", self.ablation_seed,
                            self.itr, self._num_steps, purpose)

    def _take_step(self, indices, context):
        # The real path consumes only the real training stream. All auxiliary
        # sampling is performed regardless of masks and restores this stream.
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        outputs, task_z, _ = self.agent(obs, context, task_indices=indices)
        new_actions = outputs[0]
        t, b, _ = obs.shape
        n_real = t * b
        with torch.no_grad():
            next_actions = self.agent.get_target_policy_action(next_obs, context, task_indices=indices)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
        obs, actions, next_obs = [x.reshape(n_real, -1) for x in (obs, actions, next_obs)]
        next_actions = next_actions.reshape(n_real, -1)
        real_z = task_z.detach()
        context_size = context.shape[1]
        target = context[..., self.obs_dim + self.action_dim:]
        expanded_z = self.agent.z.unsqueeze(1).expand(-1, context_size, -1)
        predicted = self.context_decoder(context[..., :self.obs_dim],
                                         context[..., self.obs_dim:self.obs_dim + self.action_dim], expanded_z)
        recon_loss = (predicted - target).square().mean()

        if obs.is_cuda:
            torch.cuda.synchronize(obs.device)
        generation_start = time.perf_counter()
        with self.auxiliary_rng("virtual_candidates"), torch.no_grad():
            virtual_z = self._sample_virtual_task_embeddings(context_size)
        if obs.is_cuda:
            torch.cuda.synchronize(obs.device)
        generation_seconds = time.perf_counter() - generation_start
        n_virtual = 0 if virtual_z is None else len(virtual_z)
        schedule = self._get_virtual_transition_base_loss_weight()
        with self.auxiliary_rng("virtual_transitions"):
            added, add_stats = self.add_virtual_transitions_to_buffer(virtual_z, context_size) if schedule > 0 else (0, {})
        add_stats = {key: value for key, value in add_stats.items() if "cycle_error" not in key and "nearest_real_dist" not in key}

        # One context construction and one re-encoding before any masks. The
        # full denominator uses actual generated elements, including masked V.
        recovery_z = self.agent.z_means.detach()
        anchors = np.asarray(indices)
        with self.auxiliary_rng("recovery"):
            if n_virtual:
                recovery_z = torch.cat([recovery_z, virtual_z.detach()], dim=0)
                anchors = np.concatenate([anchors, np.random.choice(self.train_tasks, n_virtual, replace=True)])
            synthetic = self._build_consistency_context(recovery_z, context_size, anchor_task_indices=anchors)
            recovered = self._get_context_embedding(synthetic.detach(), sample=False)
        recovery_loss, recovery_stats = recovery_terms(
            recovered, recovery_z, len(indices), self.mask_r, self.mask_v, self.consistency_loss_weight)
        encoder_loss = self.recon_loss_weight * recon_loss + recovery_loss
        gradient_stats = {}
        if getattr(self, "audit_gradients", False):
            parameters = list(self.agent.context_encoder.parameters())
            gradient_stats = {
                "base_encoder_gradient_norm": self._gradient_norm(self.recon_loss_weight * recon_loss, parameters),
                "R_encoder_gradient_norm": self._gradient_norm(recovery_stats["recovery_real_weighted"], parameters),
                "V_encoder_gradient_norm": self._gradient_norm(recovery_stats["recovery_virtual_weighted"], parameters),
            }
        self.context_optimizer.zero_grad()
        encoder_loss.backward()
        self.context_optimizer.step()

        # Keep virtual work identical in P=0/P=1. Mask the objective, not the
        # generator, buffer, RNG, or sample count.
        virtual = None
        with self.auxiliary_rng("virtual_rl"):
            if schedule > 0:
                virtual = self.sample_virtual_sac(self.virtual_transition_batch_size)
            if virtual is not None:
                vo, va, vr, vn, vt, vz, quality, _, _ = [x.detach() for x in virtual]
                m = vz.shape[0]
                vo, va, vn = [x.reshape(m, -1) for x in (vo, va, vn)]
                vr, vt = vr.reshape(m, 1), vt.reshape(m, 1)
                with torch.no_grad():
                    vnext_action = self.agent.get_target_policy_action(vn[None], None, given_z=vz)
                    noise = (torch.randn_like(vnext_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                    vnext_action = (vnext_action + noise).clamp(-self.max_action, self.max_action)
                vnew_action = self.agent.policy(1, m, torch.cat([vo, vz], -1),
                                                reparameterize=True, return_log_prob=True)[0]
            else:
                m = 0
                quality = obs.new_empty((0, 1))

        q1 = self.qf1(1, n_real, obs, actions, real_z)
        q2 = self.qf2(1, n_real, obs, actions, real_z)
        with torch.no_grad():
            tq = torch.minimum(self.target_qf1(1, n_real, next_obs, next_actions, real_z),
                               self.target_qf2(1, n_real, next_obs, next_actions, real_z))
            target_q = rewards.reshape(n_real, 1) * self.reward_scale + (1 - terms.reshape(n_real, 1)) * self.discount * tq
        real_errors = (q1 - target_q).square() + (q2 - target_q).square()
        virtual_errors = obs.new_empty((0, 1))
        if m:
            vq1, vq2 = self.qf1(1, m, vo, va, vz), self.qf2(1, m, vo, va, vz)
            with torch.no_grad():
                vtq = torch.minimum(self.target_qf1(1, m, vn, vnext_action, vz),
                                    self.target_qf2(1, m, vn, vnext_action, vz))
                vtarget = vr * self.reward_scale + (1 - vt) * self.discount * vtq
            virtual_errors = (vq1 - vtarget).square() + (vq2 - vtarget).square()
        critic_loss, real_critic, virtual_critic, virtual_critic_contribution = critic_terms(
            real_errors, virtual_errors, quality, schedule, self.mask_p)
        if getattr(self, "audit_gradients", False):
            gradient_stats["P_critic_gradient_norm"] = self._gradient_norm(virtual_critic_contribution, list(self.qf1.parameters()) + list(self.qf2.parameters()))
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        critic_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        self._set_requires_grad(self.qf1, False)
        self._set_requires_grad(self.qf2, False)
        q = self._min_q(1, n_real, obs, new_actions, real_z)
        vq, vbc = obs.new_empty((0, 1)), obs.new_empty((0, 1))
        if m:
            vq = self._min_q(1, m, vo, vnew_action, vz)
            vbc = (vnew_action - va).square().mean(dim=-1, keepdim=True)
        real_bc = (new_actions - actions).square().mean(dim=-1, keepdim=True)
        actor_loss, actor_stats = actor_terms(q, vq, real_bc, vbc, quality, schedule,
                                             self.mask_p, self.bc_weight, self.virtual_transition_train_policy_bc)
        if getattr(self, "audit_gradients", False):
            gradient_stats["P_actor_Q_gradient_norm"] = self._gradient_norm(actor_stats["actor_virtual_q_weighted"], self.agent.policy.parameters())
            gradient_stats["P_actor_BC_gradient_norm"] = self._gradient_norm(actor_stats["actor_virtual_bc_weighted"], self.agent.policy.parameters())
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()
        self._set_requires_grad(self.qf1, True)
        self._set_requires_grad(self.qf2, True)
        for source, target_net in ((self.qf1, self.target_qf1), (self.qf2, self.target_qf2),
                                   (self.agent.policy, self.agent.target_policy)):
            self._update_target_network(source, target_net)
        self.update_counts["encoder"] += 1
        self.update_counts["critic"] += 1
        self.update_counts["actor"] += 1
        self.update_counts["target_events"] += 1
        self.update_counts["target_networks"] += 3

        quality_mean = quality.mean().item() if m else 0.0
        stats = {
            **recovery_stats, **actor_stats, **add_stats, **gradient_stats,
            "mask_R": self.mask_r, "mask_V": self.mask_v, "mask_P": self.mask_p,
            "recon_loss": recon_loss, "context_loss": self.recon_loss_weight * recon_loss,
            "consistency_loss": recovery_loss, "encoder_total_loss": encoder_loss,
            "qf_loss": critic_loss, "real_qf_loss": real_critic,
            "virtual_qf_loss_raw": virtual_critic, "virtual_qf_loss_weighted": virtual_critic_contribution,
            "policy_total_loss": actor_loss, "virtual_requested_tasks": self.n_vt,
            "virtual_generated_tasks": n_virtual, "virtual_rejected_tasks": self.n_vt - n_virtual,
            "virtual_generation_coverage": n_virtual / self.n_vt if self.n_vt else 0.0,
            "virtual_generation_seconds": generation_seconds,
            "num_virtual_transitions_added": added, "virtual_transition_batch_size": m,
            "virtual_transition_scheduled_weight": schedule,
            "virtual_transition_quality_weight": quality_mean,
            "virtual_transition_effective_weight": self.mask_p * schedule * quality_mean,
            "virtual_transition_used_batch_size": m if self.mask_p and schedule > 0 else 0,
            "virtual_transition_policy_bc_batch_size": m if self.mask_p and schedule > 0 and self.virtual_transition_train_policy_bc else 0,
            "virtual_transition_quality_ess": quality.sum().square() / quality.square().sum().clamp(min=1e-12) if m else 0.0,
        }
        for key, value in stats.items():
            number = value.detach().item() if torch.is_tensor(value) else value
            if not np.isfinite(number):
                raise FloatingPointError("Nonfinite matched training metric: " + key)
            self.loss[key] = number
        if not hasattr(self, "usage_totals"):
            self.usage_totals = {}
        for key, value in {
            "scheduled_weight_sum": schedule, "effective_weight_sum": self.mask_p * schedule * quality_mean,
            "requested_tasks": self.n_vt, "generated_tasks": n_virtual, "transitions_added": added,
            "sampled_virtual_transitions": m, "used_virtual_transitions": self.loss["virtual_transition_used_batch_size"],
            "actor_virtual_bc_samples": self.loss["virtual_transition_policy_bc_batch_size"],
            "generation_seconds": generation_seconds,
        }.items():
            self.usage_totals[key] = self.usage_totals.get(key, 0) + value
        self.eval_statistics = OrderedDict((k, self.loss[k]) for k in ("recon_loss", "consistency_loss", "qf_loss", "policy_total_loss"))
        return np.asarray(self.agent.z_means.detach().cpu().tolist()), np.asarray(self.agent.z_vars.detach().cpu().tolist())
