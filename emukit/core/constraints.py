import logging
from typing import Callable, Optional

import numpy as np

_log = logging.getLogger(__name__)


class IConstraint:
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Array of shape (n_points x n_dims) containing input locations to evaluate constraint at
        :return: Numpy array of shape (n_input,) where an element will be 1 if the corresponding input satisfies the
                 constraint and zero if the constraint is violated
        """
        raise NotImplementedError


class InequalityConstraint(IConstraint):
    def __init__(self, lower_bound: np.ndarray, upper_bound: np.ndarray):
        """
        :param lower_bound: Lower bound vector of size (n_constraint,). Can be -np.inf for one sided constraint
        :param upper_bound: Upper bound vector of size (n_constraint,). Can be np.inf for one sided constraint
        """
        if (lower_bound is None) and (upper_bound is None):
            raise ValueError('Neither lower nor upper bounds is set, at least one must be specified')

        # Default lower bound to -infinity
        if lower_bound is None:
            lower_bound = np.full([upper_bound.shape[0]], -np.inf)

        # Default upper bound to +infinity
        if upper_bound is None:
            upper_bound = np.full([lower_bound.shape[0]], np.inf)

        if np.any((lower_bound == -np.inf) & (upper_bound == np.inf)):
            raise ValueError('One or more inequality constraints are unbounded')

        if np.any(lower_bound >= upper_bound):
            raise ValueError('Lower bound is greater than or equal to upper bound for one or more constraints')

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class LinearInequalityConstraint(InequalityConstraint):
    """
    Constraint of the form lower_bound <= Ax <= upper_bound where the matrix A is called "constraint_matrix"
    """
    def __init__(self, constraint_matrix: np.ndarray, lower_bound: np.ndarray=None, upper_bound: np.ndarray=None):
        """

        :param constraint_matrix: (n_constraint, n_x_dims) matrix in b_lower <= Ax <= b_upper
        :param lower_bound: Lower bound vector of size (n_constraint,). Can be -np.inf for one sided constraint
        :param upper_bound: Upper bound vector of size (n_constraint,). Can be np.inf for one sided constraint
        """
        super().__init__(lower_bound, upper_bound)
        if (constraint_matrix.shape[0] != lower_bound.shape[0]) or (constraint_matrix.shape[0] != upper_bound.shape[0]):
            raise ValueError('Shape mismatch between constraint matrix {} and lower {} or upper {} bounds'.format(
                              constraint_matrix.shape, lower_bound.shape, upper_bound.shape))

        self.constraint_matrix = constraint_matrix

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate whether constraints are violated or satisfied at a set of x locations

        :param x: Array of shape (n_points x n_dims) containing input locations to evaluate constraint at
        :return: Numpy array of shape (n_points, ) where an element will be 1 if the corresponding input satisfies all
                 constraints and zero if any constraint is violated
        """
        if self.constraint_matrix.shape[1] != x.shape[1]:
            raise ValueError('Dimension mismatch between constraint matrix (second dim {})' +
                            ' and input x (second dim {})'.format(self.constraint_matrix.shape[1], x.shape[1]))

        # Transpose here is needed to handle input dimensions
        # that is, A is (n_const, n_dims) and x is (n_points, n_dims)
        ax = self.constraint_matrix.dot(x.T).T
        return np.all((ax >= self.lower_bound) & (ax <= self.upper_bound), axis=1)


class NonlinearInequalityConstraint(InequalityConstraint):
    """
    Constraint of the form lower_bound <= g(x) <= upper_bound
    """
    def __init__(self, constraint_function: Callable, lower_bound: np.ndarray, upper_bound: np.ndarray,
                 jacobian_fun: Optional[Callable]=None):
        """
        :param constraint_function: function defining constraint in b_lower <= fun(x) <= b_upper.
                                    Has signature f(x) -> array, shape(m,) where x is 1d and m is the number of constraints
        :param lower_bound: Lower bound vector of size (n_constraint,). Can be -np.inf for one sided constraint
        :param upper_bound: Upper bound vector of size (n_constraint,). Can be np.inf for one sided constraint
        :param jacobian_fun: Function returning the jacobian of the constraint function. Optional, if not supplied
                             the optimizer will use finite differences to calculate the gradients of the constraint
        """

        super().__init__(lower_bound, upper_bound)

        self.fun = constraint_function
        self.jacobian_fun = jacobian_fun

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate whether constraints are violated or satisfied at a set of x locations

        :param x: Array of shape (n_points x n_dims) containing input locations to evaluate constraint at
        :return: Numpy array of shape (n_input,) where an element will be 1 if the corresponding input satisfies the
                 constraint and zero if the constraint is violated
        """
        fun_x = np.array([self.fun(x) for x in x])
        return np.all([fun_x >= self.lower_bound, fun_x <= self.upper_bound], axis=0)
