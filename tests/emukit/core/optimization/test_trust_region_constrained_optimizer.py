import pytest

import numpy as np

from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.optimization.optimizer import OptTrustRegionConstrained, apply_optimizer
from emukit.core.constraints import LinearInequalityConstraint


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


@pytest.fixture
def trust_region_constr_linear_constraint():
    constraints = [LinearInequalityConstraint(np.array([[0, 1]]), np.array([0.5]), np.array([np.inf]))]
    return OptTrustRegionConstrained([(-1, 1), (-1, 1)], constraints, 1000)


def test_trust_region_constrained_no_context(trust_region_constr_linear_constraint, objective, space):
    x0 = np.array([1, 1])
    x, f = apply_optimizer(trust_region_constr_linear_constraint, x0, space, objective, None, None, None)
    print(x)
    assert np.all(np.isclose(x, np.array([0, 0.5])))


def test_trust_region_constrained_no_context(trust_region_constr_linear_constraint, objective, space):
    x0 = np.array([1, 1])
    x, f = apply_optimizer(trust_region_constr_linear_constraint, x0, space, objective, None, None, None)
    print(x)
    assert np.all(np.isclose(x, np.array([0, 0.5])))
