import logging
from typing import Union, Callable, Optional

import numpy as np
import scipy.optimize

_log = logging.getLogger(__name__)


class IConstraint:
    def evaluate(self, x: np.array) -> np.array:
        """
        :param x: Input locations to evaluate constraint at
        :return: Numpy array of shape (n_input,) where an element will be 1 if the corresponding input satisfies the
                 constraint and zero if the constraint is violated
        """
        raise NotImplementedError

    def get_scipy_constraint(self) -> Union[scipy.optimize.LinearConstraint, scipy.optimize.NonlinearConstraint]:
        """
        Returns scipy constraint object
        :return:
        """
        raise NotImplementedError


class LinearInequalityConstraint(IConstraint):
    """
    Constraint of the form b_lower < Ax < b_upper
    """
    def __init__(self, A: np.array, b_lower: np.array, b_upper: np.array):
        """

        :param A: (n_constraint, n_x_dims) matrix in b_lower < Ax < b_upper
        :param b_lower: Lower bound vector of size (n_constraint,). Can be -np.inf for one sided constraint
        :param b_upper: Upper bound vector of size (n_constraint,). Can be np.inf for one sided constraint
        """
        self.A = A
        self.b_lower = b_lower
        self.b_upper = b_upper

    def get_scipy_constraint(self) -> scipy.optimize.LinearConstraint:
        """
        :return: A scipy LinearConstraint object for use with the scipy.optmize.minimize method
        """
        return scipy.optimize.LinearConstraint(self.A, self.b_lower, self.b_upper)

    def evaluate(self, x):
        """
        Evaluate whether constraints are violated or satisfied at a set of x locations

        :param x: Input locations to evaluate constraint at
        :return: Numpy array of shape (n_input,) where an element will be 1 if the corresponding input satisfies the
                 constraint and zero if the constraint is violated
        """
        ax = self.A.dot(x)
        return np.all([ax >= self.b_lower, ax <= self.b_upper], axis=0)


class NonlinearInequalityConstraint:
    """
    Constraint of the form b_lower < g(x) < b_upper
    """
    def __init__(self, fun: Callable, b_lower: np.array, b_upper: np.array, jacobian_fun: Optional[Callable]=None):
        """
        :param fun: function defining constraint in b_lower < fun(x) < b_upper. Has signature f(x) -> array, shape(m,)
                    where x is 1d and m is the number of constraints
        :param b_lower: Lower bound vector of size (n_constraint,). Can be -np.inf for one sided constraint
        :param b_upper: Upper bound vector of size (n_constraint,). Can be np.inf for one sided constraint
        :param jacobian_fun: Function describing
        """
        self.fun = fun
        self.b_lower = b_lower
        self.b_upper = b_upper
        self.jacobian_fun = jacobian_fun if jacobian_fun else '2-point'

    def get_scipy_constraint(self) -> scipy.optimize.NonlinearConstraint:
        """
        :return: a scipy NonlinearConstraint object for use with the scipy.optmize.minimize method
        """
        return scipy.optimize.NonlinearConstraint(self.fun, self.b_lower, self.b_upper, self.jacobian_fun)

    def evaluate(self, X):
        """
        Evaluate whether constraints are violated or satisfied at a set of x locations

        :param X: Input locations to evaluate constraint at
        :return: Numpy array of shape (n_input,) where an element will be 1 if the corresponding input satisfies the
                 constraint and zero if the constraint is violated
        """
        fun_x = np.array([self.fun(x) for x in X])
        return np.all([fun_x >= self.b_lower, fun_x <= self.b_upper], axis=0)
