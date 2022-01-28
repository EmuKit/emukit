import numpy as np
import pytest

from emukit.core.constraints import InequalityConstraint, LinearInequalityConstraint, NonlinearInequalityConstraint


@pytest.fixture
def linear_constraint():
    A = np.array([[1.0, 0.5]])
    b_lower = np.array([-5])
    b_upper = np.array([5])
    return LinearInequalityConstraint(A, b_lower, b_upper)


@pytest.fixture
def linear_constraint_2d():
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    b_lower = np.array([-5, -5])
    b_upper = np.array([5, 5])
    return LinearInequalityConstraint(A, b_lower, b_upper)


@pytest.fixture
def nonlinear_constraint_no_jac():
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    b_lower = np.array([-1])
    b_upper = np.array([1])
    return NonlinearInequalityConstraint(fun, b_lower, b_upper)


def test_linear_inequality_constraint_input_dims_mismatch(linear_constraint):
    x_test = np.array([[1], [2]])
    with pytest.raises(ValueError):
        linear_constraint.evaluate(x_test)


def test_linear_inequality_constraint_bounds_dims_mismatch():
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    b_lower = np.array([-5, -5])
    b_upper = np.array([5])
    with pytest.raises(ValueError):
        LinearInequalityConstraint(A, b_lower, b_upper)


def test_linear_inequality_constraint_satisfied(linear_constraint):
    x_test = np.array([[1, 1], [2, 2], [3, 3]])
    assert np.all(linear_constraint.evaluate(x_test))


def test_linear_inequality_constraint_satisfied_2d(linear_constraint_2d):
    x_test = np.array([[1, 1], [2, 2], [3, 3]])
    assert np.all(linear_constraint_2d.evaluate(x_test))


def test_linear_inequality_constraint_violated(linear_constraint):
    x_test = np.array([[50, 50], [20, 20]])
    assert np.all(linear_constraint.evaluate(x_test)) == False


def test_linear_inequality_constraint_violated_2d(linear_constraint_2d):
    x_test = np.array([[1, 1], [20, 20]])
    assert np.all(linear_constraint_2d.evaluate(x_test)) == False


def test_nonlinear_inequality_constraint_satisfied(nonlinear_constraint_no_jac):
    assert nonlinear_constraint_no_jac.evaluate(np.array([[0, 0]])) == 1


def test_nonlinear_inequality_constraint_violated(nonlinear_constraint_no_jac):
    assert nonlinear_constraint_no_jac.evaluate(np.array([[1, 1]])) == 0


def test_inequality_constraint_no_bounds():
    with pytest.raises(ValueError):
        InequalityConstraint(None, None)


def test_inequality_constraint_all_inf_bound():

    lower = np.array([-np.inf, 0])
    upper = np.array([np.inf, 1])
    with pytest.raises(ValueError):
        InequalityConstraint(lower, upper)


def test_inequality_constraint_unbounded():

    lower = np.array([-np.inf, 0])
    upper = None
    with pytest.raises(ValueError):
        InequalityConstraint(lower, upper)


def test_inequality_constraint_lower_bound_above_upper():
    lower = np.array([1, 0])
    upper = np.array([0, 1])
    with pytest.raises(ValueError):
        InequalityConstraint(lower, upper)
