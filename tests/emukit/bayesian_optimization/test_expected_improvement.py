import numpy as np
import pytest

from scipy.optimize import check_grad

from GPy.models import GPRegression
from GPy.kern import RBF

from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement


@pytest.fixture
def acquisition():
    rng = np.random.RandomState(42)
    x_init = rng.rand(5, 2)
    y_init = rng.rand(5, 1)
    model = GPRegression(x_init, y_init, RBF(2))
    return ExpectedImprovement(GPyModelWrapper(model))


def test_expected_improvement_shape(acquisition):

    rng = np.random.RandomState(43)

    x_test = rng.rand(10, 2)
    result = acquisition.evaluate(x_test)

    assert(result.shape == (10, 1))


def test_expected_improvement_gradient_computation(acquisition):
    rng = np.random.RandomState(43)
    x_test = rng.rand(10, 2)

    _, grad = acquisition.evaluate_with_gradients(x_test)

    assert(grad.shape == (10, 2))

    def wrapper(x):
        return acquisition.evaluate(np.array([x]))[0][0]

    def wrapper_grad(x):
        grad = acquisition.evaluate_with_gradients(np.array([x]))[1]
        return grad[0]

    for xi in x_test:
        err = check_grad(wrapper, wrapper_grad, xi, epsilon=1e-8)

        assert err < 1e-7
