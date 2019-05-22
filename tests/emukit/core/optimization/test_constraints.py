import numpy as np
import pytest

from emukit.core.optimization.constraints import (LinearInequalityConstraint,
                                                  NonlinearInequalityConstraint)


@pytest.fixture
def linear_constraint():
    A = np.array([[1., 0.5]])
    b_lower = np.array([-2])
    b_upper = np.array([3])
    return LinearInequalityConstraint(A, b_lower, b_upper)


@pytest.fixture
def nonlinear_constraint_no_jac():
    fun = lambda x: x[0]**2 + x[1]**2
    b_lower = np.array([-1])
    b_upper = np.array([1])
    return NonlinearInequalityConstraint(fun, b_lower, b_upper)


def test_linear_inequality_constraint_satisfied(linear_constraint):
    x_test = np.array([[1], [2]])
    assert linear_constraint.evaluate(x_test)[0] == 1


def test_linear_inequality_constraint_violated(linear_constraint):
    x_test = np.array([[50], [20]])
    assert linear_constraint.evaluate(x_test)[0] == 0


def test_get_scipy_linear_constraint(linear_constraint):
    scipy_constraint = linear_constraint.get_scipy_constraint()
    assert np.array_equal(scipy_constraint.A, np.array([[1., 0.5]]))
    assert np.array_equal(scipy_constraint.lb, np.array([-2]))
    assert np.array_equal(scipy_constraint.ub, np.array([3]))


def test_nonlinear_inequality_constraint_satisfied(nonlinear_constraint_no_jac):
    assert nonlinear_constraint_no_jac.evaluate(np.array([[0, 0]])) == 1


def test_nonlinear_inequality_constraint_violated(nonlinear_constraint_no_jac):
    assert nonlinear_constraint_no_jac.evaluate(np.array([[1, 1]])) == 0


def test_get_scipy_nonlinear_constraint(nonlinear_constraint_no_jac):
    nonlinear_constraint_no_jac.get_scipy_constraint()
