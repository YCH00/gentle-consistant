"""Behavioral checks for task paths, independent of Gym and MuJoCo.

Run with the project's PyTorch environment:
    python -m pytest tests/test_semantic_task_interpolation.py -q
"""

import copy
import json
from pathlib import Path

import pytest
import torch

from configs.default import default_config
from rlkit.torch.task_interpolation import SemanticTaskInterpolator


ROOT = Path(__file__).resolve().parents[1]


class CurvedRewardDecoder(torch.nn.Module):
    """A known nonlinear task chart with a shorter curved latent path.

    For endpoints (-1, 0), (1, 0), the linear path changes both the
    linear and quadratic reward coefficients.  Bending upward keeps the
    quadratic coefficient constant while the first changes monotonically.
    """

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, observations, actions, embeddings):
        return self.scale * (
            embeddings[..., :1] * observations
            + (embeddings[..., 1:2] + embeddings[..., :1].square())
            * actions
        )


class DeltaDecoder(torch.nn.Module):
    """The production decoder contract returns an absolute next state."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, observations, actions, embeddings):
        rewards = self.scale * embeddings[..., :1] * actions
        delta = embeddings[..., :1] * actions + embeddings[..., 1:2]
        return torch.cat((rewards, observations + delta), dim=-1)


class CurvedDeltaDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, observations, actions, embeddings):
        delta = self.scale * (
            embeddings[..., :1] * actions
            + (embeddings[..., 1:2] + embeddings[..., :1].square())
            * actions.square()
        )
        return torch.cat((torch.zeros_like(delta), observations + delta), dim=-1)


def paired_batches(decoder, task_z, probe_count=64, shifts=None, dynamics=False):
    """Exact observations under each toy task, with identifiable tuple rows."""
    task_count = len(task_z)
    # Both halves cover the same distribution, as independent replay draws
    # would. Chronologically splitting a sorted grid instead tests a deliberate
    # out-of-support validation bank, which should correctly be rejected.
    order = torch.randperm(probe_count)
    positions = order.to(dtype=task_z.dtype)
    observations = torch.linspace(-2.0, 2.0, probe_count)[order].reshape(1, -1, 1)
    observations = observations.repeat(task_count, 1, 1)
    if shifts is not None:
        observations = observations + torch.as_tensor(shifts).reshape(-1, 1, 1)
    actions = torch.cos(positions * 0.7).reshape(1, -1, 1)
    actions = actions.repeat(task_count, 1, 1)
    embeddings = task_z[:, None, :].expand(-1, probe_count, -1)
    with torch.no_grad():
        prediction = decoder(observations, actions, embeddings)
    if dynamics:
        next_observations = prediction[..., 1:].clone()
    else:
        # Distinctive successors catch accidental independent tuple sampling.
        next_observations = (
            observations + 0.2 * actions + torch.arange(task_count)[:, None, None]
        )
    return {
        "observations": observations,
        "actions": actions,
        "rewards": prediction[..., :1].clone(),
        "next_observations": next_observations,
        "terminals": (positions.remainder(3) == 0).to(task_z.dtype)
        .reshape(1, -1, 1)
        .repeat(task_count, 1, 1),
    }


def sampler_options(**overrides):
    options = {
        key: value
        for key, value in default_config["algo_params"].items()
        if key.startswith("virtual_semantic_")
    }
    options.update(
        virtual_semantic_neighbors=1,
        virtual_semantic_probe_batch_size=32,
        virtual_semantic_min_shared_probes=8,
        virtual_semantic_support_radius=0.6,
        virtual_semantic_path_nodes=9,
        virtual_semantic_path_steps=50,
        virtual_semantic_path_lr=0.05,
        virtual_semantic_trust_radius=0.5,
        virtual_semantic_max_reconstruction_error=0.1,
        virtual_semantic_latent_weight=0.01,
    )
    options.update(overrides)
    return options


def make_reward_sampler(task_z=None, **options):
    if task_z is None:
        task_z = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    decoder = CurvedRewardDecoder()
    batches = paired_batches(decoder, task_z)
    sampler = SemanticTaskInterpolator(decoder, False, **sampler_options(**options))
    return sampler, decoder, task_z, batches


def test_unshared_offline_support_rejects_edges():
    sampler, decoder, task_z, _ = make_reward_sampler()
    batches = paired_batches(decoder, task_z, shifts=[-100.0, 100.0])
    sampler.refresh(task_z, batches)
    assert sampler.sample(4, 16) is None


def test_unreliable_endpoint_reconstruction_rejects_edges():
    sampler, _, task_z, batches = make_reward_sampler()
    batches["rewards"] = batches["rewards"] + 100.0
    sampler.refresh(task_z, batches)
    assert sampler.sample(4, 16) is None


def test_sample_keeps_real_transition_tuples_together():
    torch.manual_seed(19)
    sampler, _, task_z, batches = make_reward_sampler()
    sampler.refresh(task_z, batches)
    sample = sampler.sample(12, 24)
    assert sample is not None
    fields = ("observations", "actions", "next_observations", "terminals")
    originals = torch.cat([batches[key] for key in fields], dim=-1).reshape(-1, 4)
    sampled = torch.cat([sample[key] for key in fields], dim=-1).reshape(-1, 4)
    matches = torch.isclose(sampled[:, None, :], originals[None, :, :], atol=1e-6)
    assert matches.all(dim=-1).any(dim=-1).all()
    assert torch.isfinite(sample["embeddings"]).all()
    assert torch.isfinite(sample["quality_weights"]).all()
    assert (sample["quality_weights"] >= 0).all()
    assert (sample["quality_weights"] <= 1).all()


def test_seed_reproduces_refresh_and_sample():
    results = []
    for _ in range(2):
        torch.manual_seed(314)
        sampler, _, task_z, batches = make_reward_sampler()
        sampler.refresh(task_z, batches)
        result = sampler.sample(8, 16)
        assert result is not None
        results.append(result)
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        if torch.is_tensor(results[0][key]):
            torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)


def test_outer_no_grad_preserves_decoder_parameters_and_existing_gradients():
    sampler, decoder, task_z, batches = make_reward_sampler()
    decoder.train()
    decoder.scale.grad = torch.tensor(7.0)
    state_before = copy.deepcopy(decoder.state_dict())
    with torch.no_grad():
        sampler.refresh(task_z, batches)
        sample = sampler.sample(4, 16)
    assert sample is not None
    assert decoder.training
    assert decoder.scale.requires_grad
    torch.testing.assert_close(decoder.scale.grad, torch.tensor(7.0), rtol=0, atol=0)
    for key, value in decoder.state_dict().items():
        torch.testing.assert_close(value, state_before[key], rtol=0, atol=0)


def test_dynamics_decoder_accepts_absolute_next_observations():
    torch.manual_seed(12)
    decoder = DeltaDecoder()
    task_z = torch.tensor([[-0.5, 0.3], [0.5, 0.7]])
    batches = paired_batches(decoder, task_z, dynamics=True)
    sampler = SemanticTaskInterpolator(
        decoder,
        True,
        **sampler_options(virtual_semantic_dynamics_weight=1.0),
    )
    sampler.refresh(task_z, batches)
    sample = sampler.sample(5, 16)
    assert sample is not None
    for key in ("embeddings", "quality_weights", "next_observations"):
        assert torch.isfinite(sample[key]).all()


def test_dynamics_geometry_uses_changes_not_absolute_state_variance():
    # Changing the scale of current states leaves the task-dependent deltas
    # untouched. Absolute-next-state normalization would alter this geometry.
    sampled_embeddings = []
    for observation_scale in (1.0, 30.0):
        torch.manual_seed(81)
        decoder = CurvedDeltaDecoder()
        task_z = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        batches = paired_batches(decoder, task_z, dynamics=True)
        batches["observations"] *= observation_scale
        embeddings = task_z[:, None, :].expand(-1, 64, -1)
        with torch.no_grad():
            batches["next_observations"] = decoder(
                batches["observations"], batches["actions"], embeddings
            )[..., 1:]
        sampler = SemanticTaskInterpolator(
            decoder,
            True,
            **sampler_options(
                virtual_semantic_reward_weight=0.0,
                virtual_semantic_dynamics_weight=1.0,
                virtual_semantic_latent_weight=0.2,
            ),
        )
        sampler.refresh(task_z, batches)
        sample = sampler.sample(8, 16)
        assert sample is not None
        sampled_embeddings.append(sample["embeddings"])
    torch.testing.assert_close(
        sampled_embeddings[0], sampled_embeddings[1], rtol=2e-3, atol=2e-3
    )


def test_all_environment_profiles_have_valid_semantic_settings():
    paths = sorted((ROOT / "configs").glob("*.json"))
    assert paths
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            profile = json.load(stream)
        params = dict(default_config["algo_params"])
        params.update(profile.get("algo_params", {}))
        assert params["virtual_task_generation_mode"] == "semantic", path.name
        semantic = {key: value for key, value in params.items() if key.startswith("virtual_semantic_")}
        # Construction must validate every field, including dynamics profiles.
        SemanticTaskInterpolator(
            DeltaDecoder() if params["use_next_obs_in_context"] else CurvedRewardDecoder(),
            params["use_next_obs_in_context"],
            **semantic,
        )
        if profile.get("env_name") == "cheetah-dir":
            assert params["n_vt"] == 0, path.name
            assert params["virtual_transition_loss_weight"] == 0, path.name
        else:
            assert params["n_vt"] > 0, path.name


def test_nonlinear_refinement_improves_task_energy_with_fixed_endpoints():
    torch.manual_seed(8)
    sampler, decoder, task_z, batches = make_reward_sampler()
    sampler.refresh(task_z, batches)
    edge = sampler.edges[(0, 1)]
    nodes = edge['nodes']
    torch.testing.assert_close(nodes[0], task_z[0], rtol=0, atol=0)
    torch.testing.assert_close(nodes[-1], task_z[-1], rtol=0, atol=0)
    t = torch.linspace(0, 1, len(nodes))[:, None]
    reference = (1 - t) * task_z[0] + t * task_z[1]
    endpoint_distance = (task_z[1] - task_z[0]).norm()
    assert (nodes - reference).norm(dim=-1).max() <= 0.5 * endpoint_distance + 1e-6
    assert (nodes[1:] - nodes[:-1]).norm(dim=-1).max() <= 2 * endpoint_distance / (len(nodes) - 1) + 1e-6
    # An actual bend in the second coordinate, not merely nonuniform timing
    # along the original line, must reduce changes in the task's predictions.
    assert nodes[1:-1, 1].max() > 0.05
    probes = torch.linspace(-2, 2, 37)[None, :, None].expand(len(nodes), -1, -1)
    actions = torch.cos(torch.arange(37) * 0.7)[None, :, None].expand_as(probes)
    curved = decoder(probes, actions, nodes[:, None, :].expand(-1, 37, -1))
    straight = decoder(probes, actions, reference[:, None, :].expand(-1, 37, -1))
    assert (curved[1:] - curved[:-1]).square().sum() < (straight[1:] - straight[:-1]).square().sum()
    assert sampler.stats['virtual_semantic_refinement_accepted'] > 0


def test_disconnected_graph_never_generates_cross_component_tasks():
    task_z = torch.tensor([[-0.9, 0.0], [-0.7, 0.0], [0.7, 0.0], [0.9, 0.0]])
    sampler, _, task_z, batches = make_reward_sampler(task_z, virtual_semantic_path_steps=0)
    sampler.refresh(task_z, batches)
    assert set(sampler.edges) == {(0, 1), (2, 3)}
    sample = sampler.sample(80, 8)
    assert sample is not None
    assert sample['embeddings'][:, 0].abs().min() >= 0.7


def test_graph_paths_connect_distant_tasks_only_through_short_edges():
    task_z = torch.tensor([[-0.9, 0.0], [-0.3, 0.0], [0.3, 0.0], [0.9, 0.0]])
    sampler, _, task_z, batches = make_reward_sampler(
        task_z, virtual_semantic_neighbors=2, virtual_semantic_max_latent_distance_ratio=1.1,
        virtual_semantic_path_steps=0,
    )
    sampler.refresh(task_z, batches)
    assert (0, 3) not in sampler.edges
    assert any(path['edges'] == [(0, 1), (1, 2), (2, 3)] for path in sampler.paths)


def test_asymmetric_alpha_range_samples_near_both_endpoints():
    sampler, _, task_z, batches = make_reward_sampler(
        virtual_semantic_path_steps=0,
        virtual_semantic_alpha_min=0.01, virtual_semantic_alpha_max=0.1,
    )
    sampler.refresh(task_z, batches)
    sample = sampler.sample(100, 4)
    assert (sample['embeddings'][:, 0] < -0.5).any()
    assert (sample['embeddings'][:, 0] > 0.5).any()


def test_invalid_refinement_returns_only_the_validated_reference():
    sampler, _, task_z, batches = make_reward_sampler()

    def invalid_refinement(reference, *args):
        nodes = reference.clone()
        nodes[2] += 100
        return nodes, False

    sampler._refine = invalid_refinement
    sampler.refresh(task_z, batches)
    assert sampler.stats['virtual_semantic_reference_fallbacks'] == 1
    assert sampler.stats['virtual_semantic_refinement_accepted'] == 0
    assert sampler.sample(3, 8) is not None
    assert torch.count_nonzero(sampler.edges[(0, 1)]['nodes'][:, 1]) == 0


def test_failed_refresh_clears_previously_accepted_geometry():
    sampler, _, task_z, batches = make_reward_sampler(virtual_semantic_path_steps=0)
    sampler.refresh(task_z, batches)
    assert sampler.sample(1, 4) is not None
    batches['observations'][0, 0, 0] = float('nan')
    sampler.refresh(task_z, batches)
    assert sampler.sample(1, 4) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_sampler_runs_on_cuda_without_device_mismatch():
    sampler, decoder, task_z, batches = make_reward_sampler(virtual_semantic_path_steps=3)
    decoder.cuda()
    sampler.refresh(task_z.cuda(), {key: value.cuda() for key, value in batches.items()})
    sample = sampler.sample(3, 8)
    assert sample is not None
    assert all(value.is_cuda for value in sample.values())
    assert torch.isfinite(sample['embeddings']).all()
