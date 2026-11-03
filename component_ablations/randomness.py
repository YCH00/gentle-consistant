from contextlib import contextmanager
import random
import numpy as np
import torch
from .common import seed_for


@contextmanager
def isolated_rng(*key):
    """Counter-keyed auxiliary streams; restore real-training RNG on exit."""
    np_state, py_state = np.random.get_state(), random.getstate()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        seed = seed_for(*key)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)


def capture_rng():
    return {"numpy": np.random.get_state(), "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
