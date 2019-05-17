import logging
from typing import List, Tuple, Union, Callable, Optional

import numpy as np
import scipy.optimize

from .. import ParameterSpace
from ..acquisition import Acquisition
from .acquisition_optimizer import AcquisitionOptimizerBase
from .anchor_points_generator import AnchorPointsGenerator
from .context_manager import ContextManager
from .optimizer import apply_optimizer, Optimizer

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


class ConstrainedGradientAcquisitionOptimizer(AcquisitionOptimizerBase):
    """ Optimizes the acquisition function using a quasi-Newton method (L-BFGS).
    Can be used for continuous acquisition functions.
    """
    def __init__(self, space: ParameterSpace, constraints: List[IConstraint]) -> None:
        """
        :param space: The parameter space spanning the search problem.
        """
        super().__init__(space)
        self.constraints = constraints

    def _optimize(self, acquisition: Acquisition, context_manager: ContextManager) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of abstract method.
        Taking into account gradients if acquisition supports them.

        See AcquisitionOptimizerBase._optimizer for parameter descriptions.
        See class docstring for implementation details.
        """

        # Take negative of acquisition function because they are to be maximised and the optimizers minimise
        f = lambda x: -acquisition.evaluate(x)

        # Context validation
        if len(context_manager.contextfree_space.parameters) == 0:
            _log.warning("All parameters are fixed through context")
            x = np.array(context_manager.context_values)[None, :]
            return x, f(x)

        if acquisition.has_gradients:
            def f_df(x):
                f_value, df_value = acquisition.evaluate_with_gradients(x)
                return -f_value, -df_value
        else:
            f_df = None

        optimizer = OptTrustRegionConstrained(context_manager.contextfree_space.get_bounds(), self.constraints)

        anchor_points_generator = ConstrainedObjectiveAnchorPointsGenerator(self.space, acquisition, self.constraints)

        # Select the anchor points (with context)
        anchor_points = anchor_points_generator.get(num_anchor=1, context_manager=context_manager)

        _log.info("Starting gradient-based optimization of acquisition function {}".format(type(acquisition)))
        optimized_points = []
        for a in anchor_points:
            optimized_point = apply_optimizer(optimizer, a, f=f, df=None, f_df=f_df, context_manager=context_manager,
                                              space=self.space)
            optimized_points.append(optimized_point)

        x_min, fx_min = min(optimized_points, key=lambda t: t[1])
        return x_min, -fx_min


class OptTrustRegionConstrained(Optimizer):
    """
    Wrapper for Trust-Region Constrained algorithm that can deal with non-linear constraints
    """

    def __init__(self, bounds, constraints: List[IConstraint], max_iterations: int=1000):
        super().__init__(bounds)
        self.max_iterations = max_iterations
        self.constraints = [c.get_scipy_constraint() for c in constraints]

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        Run Trust region constrained optimization algorithm

        :param x0: Initial start point
        :param f: Function to optimize
        :param df: Derivative of function to optimize
        :param f_df: Function returning both value of objective and its gradient
        :return:
        """
        if df is None and f_df is not None:
            df_1d = lambda x: f_df(x)[1]
        elif df is not None:
            df_1d = lambda x: df(x)[0, :]

        options = {'maxiter': self.max_iterations}

        if df is None:
            res = scipy.optimize.minimize(f, x0=x0[0, :], method='trust-constr', bounds=self.bounds, jac='2-point',
                                          options=options, constraints=self.constraints, hess=scipy.optimize.BFGS())
        else:
            res = scipy.optimize.minimize(f, x0=x0[0, :], method='trust-constr', bounds=self.bounds, jac=df_1d,
                                          options=options, constraints=self.constraints, hess=scipy.optimize.BFGS())

        result_x = np.atleast_2d(res.x)
        result_fx = np.atleast_2d(res.fun)
        return result_x, result_fx


class ConstrainedObjectiveAnchorPointsGenerator(AnchorPointsGenerator):
    """
    This anchor points generator chooses points where the acquisition function is highest and the constraints are
    satisfied
    """

    def __init__(self, space: ParameterSpace, acquisition: Acquisition, constraints: List[IConstraint],
                 num_samples=1000):
        """
        :param space: The parameter space describing the input domain of the non-context variables
        :param acquisition: The acquisition function
        :param num_samples: The number of points at which the anchor point scores are calculated
        """
        super().__init__(space, num_samples)
        self.acquisition = acquisition
        self.constraints = constraints

    def get_anchor_point_scores(self, X) -> np.array:
        """
        :param X: The samples at which to evaluate the criterion
        :return: Array with score for each input point. Score is -infinity if the constraints are violated at that point
        """
        are_constraints_satisfied = np.all([c.evaluate(X) for c in self.constraints], axis=0)
        scores = np.zeros((X.shape[0],1))
        scores[~are_constraints_satisfied, 0] = -np.inf
        scores[are_constraints_satisfied, :] = self.acquisition.evaluate(X[are_constraints_satisfied, :])
        return scores
