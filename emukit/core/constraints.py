import logging
from typing import Callable, Optional

import numpy as np

_log = logging.getLogger(__name__)


class IConstraint:
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Input locations to evaluate constraint at
        :return: Numpy array of shape (n_input,) where an element will be 1 if the corresponding input satisfies the
                 constraint and zero if the constraint is violated
        """
        raise NotImplementedError


class LinearInequalityConstraint(IConstraint):
    """
    Constraint of the form lower_bound < Ax < upper_bound where the matrix A is called "constraint_matrix"
    """
    def __init__(self, constraint_matrix: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray):
        """

        :param constraint_matrix: (n_constraint, n_x_dims) matrix in b_lower < Ax < b_upper
        :param lower_bound: Lower bound vector of size (n_constraint,). Can be -np.inf for one sided constraint
        :param upper_bound: Upper bound vector of size (n_constraint,). Can be np.inf for one sided constraint
        """
        self.constraint_matrix = constraint_matrix
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate whether constraints are violated or satisfied at a set of x locations

        :param x: Input locations to evaluate constraint at
        :return: Numpy array of shape (n_input,) where an element will be 1 if the corresponding input satisfies the
                 constraint and zero if the constraint is violated
        """
        ax = self.constraint_matrix.dot(x)
        return np.all([ax >= self.lower_bound, ax <= self.upper_bound], axis=0)


class NonlinearInequalityConstraint:
    """
    Constraint of the form lower_bound < g(x) < upper_bound
    """
    def __init__(self, fun: Callable, lower_bound: np.ndarray, upper_bound: np.ndarray,
                 jacobian_fun: Optional[Callable]=None):
        """
        :param fun: function defining constraint in b_lower < fun(x) < b_upper. Has signature f(x) -> array, shape(m,)
                    where x is 1d and m is the number of constraints
        :param lower_bound: Lower bound vector of size (n_constraint,). Can be -np.inf for one sided constraint
        :param upper_bound: Upper bound vector of size (n_constraint,). Can be np.inf for one sided constraint
        :param jacobian_fun: Function describing
        """
        self.fun = fun
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.jacobian_fun = jacobian_fun if jacobian_fun else '2-point'

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate whether constraints are violated or satisfied at a set of x locations

        :param X: Input locations to evaluate constraint at
        :return: Numpy array of shape (n_input,) where an element will be 1 if the corresponding input satisfies the
                 constraint and zero if the constraint is violated
        """
        fun_x = np.array([self.fun(x) for x in X])
        return np.all([fun_x >= self.lower_bound, fun_x <= self.upper_bound], axis=0)
