import numpy as np

from emukit.examples.profet.performance_assessment import compute_ecdf, compute_ranks, compute_runtime_feval


def test_compute_runtime_feval():
    traj = np.arange(0, 10)[::-1]
    rt = compute_runtime_feval(traj, 5)

    assert rt == 5


def test_compute_ecdf():
    error = np.random.randn(100, 20, 50)
    targets = np.random.rand(100, 10)

    ecdf = compute_ecdf(error, targets)

    assert len(ecdf["x"]) == len(ecdf["y"])


def compute_ranks():
    error = np.random.rand(2, 10, 10, 20)
    ranks = compute_ranks(error)

    assert ranks.shape[0] == 2
    assert ranks.shape[1] == 20
