import pytest
import torch


@pytest.fixture(scope='session', autouse=True)
def bounded_cpu_threads():
    # Tiny synthetic networks are slower with a large BLAS thread pool.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
