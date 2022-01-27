import numpy as np
import pytest

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.optimization import ContextManager
from emukit.core.optimization.optimizer import OptLbfgs, apply_optimizer


@pytest.fixture
def lbfgs():
    return OptLbfgs(bounds=[(-1, 1), (-1, 1)], max_iterations=1000)


@pytest.fixture
def lbfgs_context():
    return OptLbfgs(bounds=[(-1, 1)], max_iterations=1000)


@pytest.fixture
def objective():
    def objective(x):
        return x[:, 0]**2 + x[:, 1]**2
    return objective


@pytest.fixture
def gradient():
    return lambda x: np.array([2 * x[:, 0],  2 * x[:, 1]])


@pytest.fixture
def space():
    return ParameterSpace([ContinuousParameter('x', -1, 1), ContinuousParameter('y', -1, 1)])


def test_lbfgs_with_gradient_no_context(lbfgs, objective, gradient, space):
    x0 = np.array([1, 1])
    x, f = apply_optimizer(lbfgs, x0, space, objective, gradient, None, None)
    assert np.all(np.isclose(x, np.array([0, 0])))


def test_lbfgs_no_gradient_no_context(lbfgs, objective, space):
    x0 = np.array([1, 1])
    x, f = apply_optimizer(lbfgs, x0, space, objective, None, None)
    assert np.all(np.isclose(x, np.array([0, 0])))


def test_lbfgs_with_gradient_and_context(lbfgs_context, objective, gradient, space):
    context = ContextManager(space, {'x': 0.5})
    x0 = np.array([1, 1])
    x, f = apply_optimizer(lbfgs_context, x0, space, objective, gradient, None, context)
    assert np.all(np.isclose(x, np.array([0.5, 0])))


def test_lbfgs_no_gradient_with_context(lbfgs_context, objective, space):
    context = ContextManager(space, {'x': 0.5})
    x0 = np.array([1, 1])
    x, f = apply_optimizer(lbfgs_context, x0, space, objective, None, None, context)
    assert np.all(np.isclose(x, np.array([0.5, 0])))
