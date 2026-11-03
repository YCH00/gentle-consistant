"""Numerical/gradient tests use tiny networks, never formal experiment scores."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn

from component_ablations import ARMS
from component_ablations.losses import recovery_terms, critic_terms, actor_terms
from component_ablations.randomness import isolated_rng, capture_rng, restore_rng
from component_ablations.step import MatchedStep
from component_ablations.state import training_state, load_training_state, atomic_torch_save


def test_recovery_full_equals_concat_loss_and_gradient():
    predicted = torch.randn(7, 5, dtype=torch.double, requires_grad=True)
    targets = torch.randn_like(predicted, requires_grad=True)
    full, stats = recovery_terms(predicted, targets, 3, 1, 1, 0.5)
    native = 0.5 * (predicted - targets.detach()).square().mean()
    assert torch.allclose(full, native)
    assert torch.allclose(torch.autograd.grad(full, predicted, retain_graph=True)[0], torch.autograd.grad(native, predicted)[0])
    assert targets.grad is None
    assert stats["lambda_R"] == 0.5 * 3 / 7
    assert stats["lambda_V"] == 0.5 * 4 / 7


def test_recovery_masks_preserve_other_gradient_and_empty_virtual_is_finite():
    predicted = torch.randn(7, 5, dtype=torch.double, requires_grad=True)
    target = torch.randn_like(predicted)
    gradients = {}
    for masks in ((1, 1), (1, 0), (0, 1), (0, 0)):
        loss, _ = recovery_terms(predicted, target, 3, *masks, 0.5)
        gradients[masks] = torch.autograd.grad(loss, predicted, retain_graph=True)[0]
    assert torch.equal(gradients[1, 1][:3], gradients[1, 0][:3])
    assert torch.equal(gradients[1, 1][3:], gradients[0, 1][3:])
    assert torch.count_nonzero(gradients[0, 0]) == 0
    loss, stats = recovery_terms(predicted[:3], target[:3], 3, 1, 1, 0.5)
    assert torch.isfinite(loss)
    assert stats["lambda_V"] == 0 and stats["recovery_virtual_skipped"] == 1


def test_P_normalization_real_coefficient_and_alpha_are_fixed():
    r = torch.tensor([[2.0], [4.0]], requires_grad=True)
    v = torch.tensor([[40.0], [90.0], [120.0]], requires_grad=True)
    quality = torch.tensor([[1.0], [0.4], [0.8]], requires_grad=True)
    gradients = []
    for mask in (0, 1):
        loss, *_ = critic_terms(r, v, quality, 0.7, mask)
        gradients.append(torch.autograd.grad(loss, (r, v), retain_graph=True))
    assert torch.equal(gradients[0][0], gradients[1][0])
    assert torch.count_nonzero(gradients[0][1]) == 0
    assert torch.count_nonzero(gradients[1][1]) > 0
    rq, vq = r * 2, v * 10
    alphas = []
    for mask in (0, 1):
        loss, stats = actor_terms(rq, vq, r.square(), v.square(), quality, 0.7, mask, 2.5, True)
        alphas.append(stats["actor_alpha"])
        assert torch.isfinite(loss)
    assert torch.equal(alphas[0], alphas[1])
    assert not alphas[0].requires_grad
    assert quality.grad is None


def test_auxiliary_rng_restores_real_stream_even_on_failure():
    np.random.seed(31)
    torch.manual_seed(31)
    expected_np = np.random.randint(0, 100, size=8)
    expected_torch = torch.rand(8)
    np.random.seed(31)
    torch.manual_seed(31)
    with pytest.raises(RuntimeError):
        with isolated_rng("test", "virtual"):
            np.random.rand(200)
            torch.rand(200)
            raise RuntimeError("test")
    assert np.array_equal(expected_np, np.random.randint(0, 100, size=8))
    assert torch.equal(expected_torch, torch.rand(8))


class TinyPolicy(nn.Linear):
    def __init__(self):
        super().__init__(3, 1)
    def forward(self, t, b, inputs, **kwargs):
        return (torch.tanh(super().forward(inputs)),)


class TinyAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_encoder = nn.Linear(3, 2)
        self.policy = TinyPolicy()
        self.target_policy = copy.deepcopy(self.policy)
        self.use_ib = False
        self.z = self.z_means = torch.zeros(2, 2)
        self.z_vars = torch.zeros(2, 2)
    def forward(self, obs, context, **kwargs):
        self.z = self.z_means = self.context_encoder(context).mean(1)
        self.z_vars = self.context_encoder(context).std(1)
        t, b, _ = obs.shape
        z = self.z[:, None].expand(-1, b, -1).reshape(t*b, 2)
        return self.policy(t, b, torch.cat([obs.reshape(t*b, 1), z.detach()], -1)), z, None
    def get_target_policy_action(self, obs, context, given_z=None, **kwargs):
        t, b, _ = obs.shape
        z = given_z if given_z is not None else self.z[:, None].expand(-1, b, -1).reshape(t*b, 2)
        return self.target_policy(t, b, torch.cat([obs.reshape(t*b, 1), z.detach()], -1))[0]


class TinyQ(nn.Linear):
    def __init__(self):
        super().__init__(4, 1)
    def forward(self, t, b, obs, action, z):
        return super().forward(torch.cat([obs, action, z], -1))


class TinyDecoder(nn.Linear):
    def __init__(self):
        super().__init__(4, 1)
        self.requires_grad_(False)
    def forward(self, obs, action, z):
        return super().forward(torch.cat([obs, action, z], -1))


class TinyTraining(MatchedStep):
    def __init__(self, masks):
        self.mask_r, self.mask_v, self.mask_p = masks
        self.ablation_seed, self.itr, self._num_steps = 0, 0, 0
        self.obs_dim, self.action_dim, self.n_vt = 1, 1, 3
        self.train_tasks, self.consistency_loss_weight = [0, 1], 0.5
        self.recon_loss_weight, self.bc_weight, self.discount, self.reward_scale = 10, 2.5, 0.99, 5
        self.policy_noise, self.noise_clip, self.max_action = 0.2, 0.5, 1.0
        self.virtual_transition_batch_size = 6
        self.virtual_transition_train_policy_bc = True
        self.agent, self.context_decoder = TinyAgent(), TinyDecoder()
        self.qf1, self.qf2 = TinyQ(), TinyQ()
        self.target_qf1, self.target_qf2 = copy.deepcopy(self.qf1), copy.deepcopy(self.qf2)
        for name, network in (("policy", self.agent.policy), ("qf1", self.qf1), ("qf2", self.qf2), ("context", self.agent.context_encoder)):
            setattr(self, name + "_optimizer", torch.optim.Adam(network.parameters(), lr=0.001))
        self.update_counts = dict(encoder=0, critic=0, actor=0, target_events=0, target_networks=0)
        self.loss, self.eval_statistics = {}, None
        self.virtual_offset, self.generated, self.virtual_inputs = 0.0, 0, []
        for name in ("_n_train_steps_total", "_n_env_steps_total", "_n_rollouts_total"):
            setattr(self, name, 0)
        for name in ("replay_buffer", "enc_replay_buffer", "relabel_buffer", "virtual_transition_buffer"):
            setattr(self, name, SimpleNamespace(_size=2, _top=2, _observations=np.ones((8, 3))))
        self._virtual_transition_recon_loss_ema, self._virtual_transition_recon_weight = None, 1.0
    def sample_sac(self, indices):
        self.real_rows = np.random.randint(0, 40, size=8)
        rows = torch.tensor(self.real_rows.tolist(), dtype=torch.float32).reshape(2, 4, 1) / 40
        return rows, rows * 0.1, rows - 0.2, rows + 0.2, torch.zeros_like(rows)
    def _sample_virtual_task_embeddings(self, batch_size):
        self.generated += self.n_vt
        return torch.rand(self.n_vt, 2)
    def _get_virtual_transition_base_loss_weight(self):
        return 0.7
    def add_virtual_transitions_to_buffer(self, z, batch_size):
        return len(z) * batch_size, {}
    @torch.no_grad()
    def _build_consistency_context(self, z, batch_size, **kwargs):
        obs, action = torch.rand(len(z), batch_size, 1), torch.rand(len(z), batch_size, 1)
        target = self.context_decoder(obs, action, z[:, None].expand(-1, batch_size, -1))
        return torch.cat([obs, action, target], -1)
    def _get_context_embedding(self, context, **kwargs):
        return self.agent.context_encoder(context).mean(1)
    def sample_virtual_sac(self, n):
        tensors = [torch.rand(1, n, 1, requires_grad=True) + self.virtual_offset for _ in range(5)]
        z = torch.rand(n, 2, requires_grad=True)
        q = torch.ones(n, 1)
        self.virtual_inputs = tensors + [z]
        for item in self.virtual_inputs:
            item.retain_grad()
        return (*tensors, z, q, q * 0, q * 0)
    def _min_q(self, t, b, obs, action, z):
        return torch.minimum(self.qf1(t, b, obs, action, z), self.qf2(t, b, obs, action, z))
    def _set_requires_grad(self, module, flag):
        module.requires_grad_(flag)
    def _update_target_network(self, network, target):
        with torch.no_grad():
            for p, tp in zip(network.parameters(), target.parameters()):
                tp.mul_(0.995).add_(p, alpha=0.005)
    def get_epoch_snapshot(self, itr):
        networks = {"qf1": self.qf1, "qf2": self.qf2, "target_qf1": self.target_qf1, "target_qf2": self.target_qf2,
                    "policy": self.agent.policy, "target_policy": self.agent.target_policy,
                    "context_encoder": self.agent.context_encoder, "context_decoder": self.context_decoder}
        return {key: network.state_dict() for key, network in networks.items()}


@pytest.mark.parametrize("arm", list(ARMS))
def test_actual_matched_step_six_arms(arm):
    torch.manual_seed(7)
    np.random.seed(7)
    model = TinyTraining(ARMS[arm])
    model.audit_gradients = True
    decoder_before = copy.deepcopy(model.context_decoder.state_dict())
    context = torch.rand(2, 4, 3)
    model._take_step([0, 1], context)
    assert model.generated == 3
    assert model.loss["virtual_generated_tasks"] == 3
    assert model.update_counts["actor"] == model.update_counts["critic"] == 1
    for parameter in model.context_decoder.parameters():
        assert parameter.grad is None
    for key, value in decoder_before.items():
        assert torch.equal(value, model.context_decoder.state_dict()[key])
    assert all(item.grad is None for item in model.virtual_inputs)
    assert sum(p.grad.abs().sum() for p in model.agent.context_encoder.parameters()) > 0
    for flag, key in ((0, "recovery_real_weighted"), (1, "recovery_virtual_weighted")):
        assert (model.loss[key] > 0) == bool(ARMS[arm][flag])
    assert (model.loss["virtual_qf_loss_weighted"] > 0) == bool(ARMS[arm][2])
    assert model.loss["virtual_transition_policy_bc_batch_size"] == (6 if ARMS[arm][2] else 0)
    for key, enabled in (("R_encoder_gradient_norm", ARMS[arm][0]), ("V_encoder_gradient_norm", ARMS[arm][1]),
                         ("P_critic_gradient_norm", ARMS[arm][2]), ("P_actor_Q_gradient_norm", ARMS[arm][2]), ("P_actor_BC_gradient_norm", ARMS[arm][2])):
        assert (model.loss[key] > 0) == bool(enabled)
    assert all(np.isfinite(value) for value in model.loss.values())


def test_P_zero_real_step_independent_of_virtual_inputs_and_rng_matches_all_arms():
    torch.manual_seed(11)
    initial = TinyTraining((0, 0, 0))
    context = torch.rand(2, 4, 3)
    snapshots, rows = [], []
    rng = capture_rng()
    for offset in (0.0, 50.0):
        model = copy.deepcopy(initial)
        model.virtual_offset = offset
        restore_rng(rng)
        model._take_step([0, 1], context)
        snapshots.append(copy.deepcopy(model.get_epoch_snapshot(0)))
    for name in snapshots[0]:
        for key in snapshots[0][name]:
            assert torch.equal(snapshots[0][name][key], snapshots[1][name][key])
    for masks in ARMS.values():
        model = copy.deepcopy(initial)
        model.mask_r, model.mask_v, model.mask_p = masks
        restore_rng(rng)
        model._take_step([0, 1], context)
        rows.append(model.real_rows)
    assert all(np.array_equal(rows[0], item) for item in rows)


def test_complete_state_resume_matches_uninterrupted_next_step(tmp_path):
    torch.manual_seed(2)
    np.random.seed(2)
    model = TinyTraining((1, 1, 1))
    context = torch.rand(2, 4, 3)
    model._take_step([0, 1], context)
    model._num_steps += 1
    state = training_state(model, {"hash": "test"}, 1)
    path = tmp_path / "resume.pt"
    atomic_torch_save(path, state)
    model._take_step([0, 1], context)
    expected = copy.deepcopy(model.get_epoch_snapshot(1))
    restored = TinyTraining((1, 1, 1))
    next_iteration = load_training_state(restored, torch.load(path, weights_only=False), {"hash": "test"})
    assert next_iteration == 1
    restored._take_step([0, 1], context)
    for name, network in expected.items():
        for key, value in network.items():
            assert torch.equal(value, restored.get_epoch_snapshot(1)[name][key])
    with pytest.raises(ValueError):
        load_training_state(restored, state, {"hash": "wrong"})
    state["buffers"] = {}
    with pytest.raises(ValueError):
        load_training_state(restored, state, {"hash": "test"})
