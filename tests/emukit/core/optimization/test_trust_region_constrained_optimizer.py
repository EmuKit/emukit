import pytest

import numpy as np

from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.optimization.optimizer import OptTrustRegionConstrained, apply_optimizer
from emukit.core.constraints import LinearInequalityConstraint, NonlinearInequalityConstraint


@pytest.fixture
def objective():
    return lambda x: (x[:, 0]**2 + x[:, 1]**2)


@pytest.fixture
def gradient():
    return lambda x: np.stack([2 * x[:, 0],  2 * x[:, 1]], axis=1)


@pytest.fixture
def space():
    return ParameterSpace([ContinuousParameter('x', -1, 1), ContinuousParameter('y', -1, 1)])


@pytest.fixture
def trust_region_constr_linear_constraint():
    constraints = [LinearInequalityConstraint(np.array([[0, 1]]), np.array([0.5]), np.array([np.inf]))]
    return OptTrustRegionConstrained([(-1, 1), (-1, 1)], constraints, 1000)


@pytest.fixture
def trust_region_constr_nonlinear_constraint():
    constraint_function = lambda x: (x[0] + x[1])**2
    constraints = [NonlinearInequalityConstraint(constraint_function, np.array([2.]), np.array([np.inf]))]
    return OptTrustRegionConstrained([(-1, 1), (-1, 1)], constraints, 1000)


def test_trust_region_constrained_no_context(trust_region_constr_linear_constraint, objective, space):
    x0 = np.array([1, 1])
    x, f = apply_optimizer(trust_region_constr_linear_constraint, x0, space, objective, None, None, None)
    assert np.all(np.isclose(x, np.array([0, 0.5])))


def test_trust_region_constrained_no_context(trust_region_constr_linear_constraint, objective, space):
    x0 = np.array([1, 1])
    x, f = apply_optimizer(trust_region_constr_linear_constraint, x0, space, objective, None, None, None)
    assert np.all(np.isclose(x, np.array([0, 0.5])))


def test_trust_region_constrained_no_context_with_gradient(trust_region_constr_linear_constraint, objective, gradient, space):
    # Tests the optimizer when passing in f and df as separate function handles
    x0 = np.array([1, 1])
    x, f = apply_optimizer(trust_region_constr_linear_constraint, x0, space, objective, gradient, None, None)
    assert np.all(np.isclose(x, np.array([0, 0.5])))


def test_trust_region_constrained_nonlinear_constraint(trust_region_constr_nonlinear_constraint, objective, gradient, space):
    # Tests the optimizer when passing in f and df as separate function handles
    x0 = np.array([1, 1])
    x, f = apply_optimizer(trust_region_constr_nonlinear_constraint, x0, space, objective, gradient, None, None)
    assert np.all(np.isclose(x, np.array([np.sqrt(2)/2, np.sqrt(2)/2]), atol=1e-3))


def test_trust_region_constrained_no_context_with_f_df(trust_region_constr_linear_constraint, objective, gradient, space):
    # Tests the optimizer when passing in f and df as a single function handle
    f_df = lambda x: (objective(x), gradient(x))
    x0 = np.array([1, 1])
    x, f = apply_optimizer(trust_region_constr_linear_constraint, x0, space, None, None, f_df, None)
    assert np.all(np.isclose(x, np.array([0, 0.5])))


def test_invalid_constraint_object():
    constraint_function = lambda x: (x[0] + x[1])**2
    with pytest.raises(ValueError):
        return OptTrustRegionConstrained([(-1, 1), (-1, 1)], [constraint_function], 1000)
