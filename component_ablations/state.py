"""Complete epoch-boundary training state; native weight snapshots are not resumes."""
import copy
from pathlib import Path
import os
import torch
from .randomness import capture_rng, restore_rng

OPTIMIZERS = ("policy_optimizer", "qf1_optimizer", "qf2_optimizer", "context_optimizer")
BUFFERS = ("replay_buffer", "enc_replay_buffer", "relabel_buffer", "virtual_transition_buffer")
COUNTERS = ("_n_train_steps_total", "_n_env_steps_total", "_n_rollouts_total", "_num_steps")


def atomic_torch_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, str(tmp))
    os.replace(str(tmp), str(path))


def pack_buffer(buffer):
    if buffer is None:
        return None
    if hasattr(buffer, "task_buffers"):
        return {"tasks": {key: pack_buffer(value) for key, value in buffer.task_buffers.items()}}
    state = {}
    for key, value in vars(buffer).items():
        if key.startswith("_") and hasattr(value, "shape"):
            state[key] = value[:buffer._size].copy()
        elif key in ("_top", "_size", "_episode_starts", "_cur_episode_start"):
            state[key] = copy.deepcopy(value)
    return state


def unpack_buffer(buffer, state):
    if state is None:
        return
    if "tasks" in state:
        for key, value in state["tasks"].items():
            unpack_buffer(buffer.task_buffers[key], value)
        return
    for key, value in state.items():
        if hasattr(value, "shape"):
            getattr(buffer, key)[:len(value)] = value
        else:
            setattr(buffer, key, copy.deepcopy(value))


def training_state(algorithm, identity, next_iteration, include_buffers=True):
    env_rng = getattr(getattr(algorithm, "env", None), "np_random", None)
    return {
        "format": "component-ablations-v1-complete-state-v1", "identity": identity,
        "next_iteration": next_iteration,
        "networks": {key: {name: tensor.detach().cpu().clone() for name, tensor in values.items()}
                     for key, values in algorithm.get_epoch_snapshot(algorithm.itr).items()},
        "optimizers": {name: getattr(algorithm, name).state_dict() for name in OPTIMIZERS},
        "rng": capture_rng(), "buffers": {name: pack_buffer(getattr(algorithm, name)) for name in BUFFERS} if include_buffers else {},
        "counters": {name: getattr(algorithm, name) for name in COUNTERS},
        "update_counts": algorithm.update_counts.copy(),
        "runtime_totals": copy.deepcopy(getattr(algorithm, "runtime_totals", {})),
        "usage_totals": copy.deepcopy(getattr(algorithm, "usage_totals", {})),
        "schedule": {"iteration": algorithm.itr, "recon_loss_ema": algorithm._virtual_transition_recon_loss_ema,
                     "recon_weight": algorithm._virtual_transition_recon_weight},
        "agent_buffers": {name: getattr(algorithm.agent, name).detach().cpu() for name in ("z", "z_means", "z_vars")},
        "env_rng": env_rng.get_state() if hasattr(env_rng, "get_state") else None,
    }


def load_training_state(algorithm, state, expected_identity, require_buffers=True):
    if state.get("format") != "component-ablations-v1-complete-state-v1" or state["identity"] != expected_identity:
        raise ValueError("Checkpoint format/config/source/resource identity mismatch")
    if require_buffers and not state["buffers"]:
        raise ValueError("Network-only snapshots cannot resume training")
    networks = {"qf1": algorithm.qf1, "qf2": algorithm.qf2, "target_qf1": algorithm.target_qf1,
                "target_qf2": algorithm.target_qf2, "policy": algorithm.agent.policy,
                "target_policy": algorithm.agent.target_policy, "context_encoder": algorithm.agent.context_encoder,
                "context_decoder": algorithm.context_decoder}
    for name, network in networks.items():
        network.load_state_dict(state["networks"][name], strict=True)
    for name in OPTIMIZERS:
        getattr(algorithm, name).load_state_dict(state["optimizers"][name])
    for name, buffer_state in state["buffers"].items():
        unpack_buffer(getattr(algorithm, name), buffer_state)
    for name, value in state["counters"].items():
        setattr(algorithm, name, value)
    algorithm.update_counts = state["update_counts"].copy()
    algorithm.runtime_totals = copy.deepcopy(state.get("runtime_totals", {}))
    algorithm.usage_totals = copy.deepcopy(state.get("usage_totals", {}))
    algorithm.itr = state["schedule"]["iteration"]
    algorithm._virtual_transition_recon_loss_ema = state["schedule"]["recon_loss_ema"]
    algorithm._virtual_transition_recon_weight = state["schedule"]["recon_weight"]
    device = next(algorithm.agent.policy.parameters()).device
    for name, value in state["agent_buffers"].items():
        setattr(algorithm.agent, name, value.to(device))
    restore_rng(state["rng"])
    if state.get("env_rng") is not None:
        algorithm.env.np_random.set_state(state["env_rng"])
    return state["next_iteration"]
