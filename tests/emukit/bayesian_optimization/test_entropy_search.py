import numpy as np
import pytest

from scipy.optimize import check_grad

from GPy.models import GPRegression
from GPy.kern import RBF

from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import EntropySearch
from emukit.bayesian_optimization.util.mcmc_sampler import AffineInvariantEnsembleSampler
from emukit.core import ContinuousParameter, ParameterSpace


@pytest.fixture
def acquisition():
    rng = np.random.RandomState(42)
    x_init = rng.rand(5, 2)
    y_init = rng.rand(5, 1)
    model = GPyModelWrapper(GPRegression(x_init, y_init, RBF(1, lengthscale=0.1)))

    space = ParameterSpace([ContinuousParameter('x1', 0, 1)])
    sampler = AffineInvariantEnsembleSampler(space)
    return EntropySearch(model, space, sampler, num_representer_points=10)


def test_entropy_search_shape(acquisition):
    rng = np.random.RandomState(43)

    x_test = rng.rand(2, 1)
    result = acquisition.evaluate(x_test)

    assert (result.shape == (2, 1))

def test_entropy_search_update_pmin(acquisition):
    logP = acquisition.update_pmin()

    assert logP.shape[0] == acquisition.num_representer_points
    # Check if representer points are inside the bounds
    assert np.all(acquisition.space.check_points_in_domain(acquisition.representer_points))


def test_innovations(acquisition):
    # Case 1: Assume no influence of test point on representer points
    acquisition.representer_points = np.array([[1.0]])
    x = np.array([[0.0]])
    dm, dv = acquisition._innovations(x)

    assert np.any(np.abs(dm) < 1e-3)
    assert np.any(np.abs(dv) < 1e-3)

    # Case 2: Test point is close to representer points
    acquisition.representer_points = np.array([[1.0]])
    x = np.array([[0.99]])
    dm, dv = acquisition._innovations(x)
    assert np.any(np.abs(dm) > 1e-3)
    assert np.any(np.abs(dv) > 1e-3)
