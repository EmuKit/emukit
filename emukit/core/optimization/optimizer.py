# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Tuple

import numpy as np
import scipy.optimize

from .. import ParameterSpace
from ..constraints import IConstraint, LinearInequalityConstraint, NonlinearInequalityConstraint
from .context_manager import ContextManager


class Optimizer(object):
    """
    Class for a general acquisition optimizer.
    """

    def __init__(self, bounds: List[Tuple]):
        """
        :param bounds: List of min/max values for each dimension of x
        """
        self.bounds = bounds

    def optimize(self, x0: np.ndarray, f: Callable=None, df: Callable=None, f_df: Callable=None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        :return: Location of optimum and value at optimum
        """
        raise NotImplementedError("The optimize method is not implemented in the parent class.")


class OptLbfgs(Optimizer):
    """
    Wrapper for l-bfgs-b to use the true or the approximate gradients.
    """

    def __init__(self, bounds, max_iterations=1000):
        super(OptLbfgs, self).__init__(bounds)
        self.max_iterations = max_iterations

    def optimize(self, x0: np.ndarray, f: Callable=None, df: Callable=None, f_df: Callable=None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        :return: Location of optimum and value at optimum
        """

        if f_df is None and df is not None: f_df = lambda x: (float(f(x)), df(x))
        if f_df is not None:
            def _f_df(x):
                return f(x), f_df(x)[1][0]

        if f_df is None and df is None:
            res = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=self.bounds, approx_grad=True, maxiter=self.max_iterations)
        else:
            res = scipy.optimize.fmin_l_bfgs_b(_f_df, x0=x0, bounds=self.bounds, maxiter=self.max_iterations)

        # We check here if the the optimizer moved. It it didn't we report x0 and f(x0) as scipy can return NaNs
        if res[2]['task'] == b'ABNORMAL_TERMINATION_IN_LNSRCH':
            result_x = np.atleast_2d(x0)
            result_fx = np.atleast_2d(f(x0))
        else:
            result_x = np.atleast_2d(res[0])
            result_fx = np.atleast_2d(res[1])

        return result_x, result_fx


def apply_optimizer(optimizer: Optimizer, x0: np.ndarray, space: ParameterSpace, f: Callable=None, df: Callable=None,
                    f_df: Callable=None, context_manager: ContextManager=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimizes f using the optimizer supplied, deals with potential context variables.

    :param optimizer: The optimizer object that will perform the optimization
    :param x0: initial point for a local optimizer (x0 can be defined with or without the context included).
    :param f: function to optimize.
    :param df: gradient of the function to optimize.
    :param f_df: returns both the function to optimize and its gradient.
    :param context_manager: If provided, x0 (and the optimizer) operates in the space without the context
    :param space: Parameter space describing input domain, including any context variables
    :return: Location of optimum and value at optimum
    """

    if context_manager is None:
        context_manager = ContextManager(space, {})

    # Compute new objective that inputs non context variables but takes into account the values of the context ones.
    # It does nothing if no context is passed
    problem = OptimizationWithContext(x0=x0, f=f, df=df, f_df=f_df, context_manager=context_manager)

    add_context = lambda x: context_manager.expand_vector(x)

    # Optimize point
    if f is None:
        f_no_context = None
    else:
        f_no_context = problem.f_no_context

    if df is None:
        df_no_context = None
    else:
        df_no_context = problem.df_no_context

    if f_df is None:
        f_df_no_context = None
    else:
        f_df_no_context = problem.f_df_no_context

    optimized_x, _ = optimizer.optimize(problem.x0_no_context, f_no_context, df_no_context, f_df_no_context)

    # Add context and round according to the type of variables of the design space
    suggested_x_with_context = add_context(optimized_x)
    suggested_x_with_context_rounded = space.round(suggested_x_with_context)

    if f is None:
        f_opt, _ = f_df(suggested_x_with_context_rounded)
    else:
        f_opt = f(suggested_x_with_context_rounded)
    return suggested_x_with_context_rounded, f_opt


class OptimizationWithContext(object):

    def __init__(self, x0: np.ndarray, f: Callable, df: Callable=None, f_df: Callable=None,
                 context_manager: ContextManager = None):
        """
        Constructor of an objective function that takes as input a vector x of the non context variables
        and returns a value in which the context variables have been fixed.
        """
        self.x0 = np.atleast_2d(x0)
        self.f = f
        self.df = df
        self.f_df = f_df
        self.context_manager = context_manager

        if not context_manager:
            self.x0_no_context = x0
            self.f_no_context = self.f
            self.df_no_context = self.df
            self.f_df_no_context = self.f_df
        else:
            self.x0_no_context = self.x0[:, self.context_manager.non_context_idxs]
            self.f_no_context = self.f_no_context
            if self.f_df is None:
                self.df_no_context = None
                self.f_df_no_context = None
            else:
                self.df_no_context = self.df_no_context
                self.f_df_no_context = self.f_df_no_context

    def f_no_context(self, x: np.ndarray) -> np.ndarray:
        """
        Wrapper of optimization objective function which deals with adding context variables to x

        :param x: Input without context variables
        """
        x = np.atleast_2d(x)
        xx = self.context_manager.expand_vector(x)
        if x.shape[0] == 1:
            return self.f(xx)[0]
        else:
            return self.f(xx)

    def df_no_context(self, x: np.ndarray) -> np.ndarray:
        """
        Wrapper of the derivative of optimization objective function which deals with adding context variables to x

        :param x: Input without context variables
        """
        x = np.atleast_2d(x)
        xx = self.context_manager.expand_vector(x)
        _, df_no_context_xx = self.f_df(xx)
        df_no_context_xx = df_no_context_xx[:, np.array(self.context_manager.non_context_idxs)]
        return df_no_context_xx

    def f_df_no_context(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper of optimization objective function and its derivative which deals with adding context variables to x

        :param x: Input without context variables
        """
        x = np.atleast_2d(x)
        xx = self.context_manager.expand_vector(x)
        f_no_context_xx, df_no_context_xx = self.f_df(xx)
        df_no_context_xx = df_no_context_xx[:, np.array(self.context_manager.non_context_idxs)]
        return f_no_context_xx, df_no_context_xx


class OptTrustRegionConstrained(Optimizer):
    """
    Wrapper for Trust-Region Constrained algorithm that can deal with non-linear constraints
    """

    def __init__(self, bounds: List[Tuple], constraints: List[IConstraint], max_iterations: int=1000):
        """
        :param bounds: List of tuples containing (lower_bound, upper_bound) for each parameter
        :param constraints: List of constraints, can contain a mix of linear and non-linear constraints
        :param max_iterations: Maximum number of iterations before the optimizer is stopped
        """
        super().__init__(bounds)
        self.max_iterations = max_iterations
        self.constraints = _get_scipy_constraints(constraints)

    def optimize(self, x0: np.ndarray, f: Callable=None, df: Callable=None, f_df: Callable=None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Trust region constrained optimization algorithm

        :param x0: Initial start point
        :param f: Function to optimize
        :param df: Derivative of function to optimize
        :param f_df: Function returning both value of objective and its gradient
        :return: Location of optimum and function value at optimum
        """

        if (f is None) and (f_df is None):
            raise ValueError('Neither f nor f_df are supplied - you must supply an objective function')

        # If f not supplied, make lambda that returns objective only from f_df
        if f is None:
            f = lambda x: f_df(x)[0]

        if df is None and f_df is not None:
            # If df not supplied and f_df is, make lambda that returns gradient only from f_df
            df_1d = lambda x: f_df(x)[1][0, :]
        elif df is not None:
            # If df is supplied, convert the 2d output to 1d
            df_1d = lambda x: df(x)[0, :]
        else:
            # Gradient not supplied
            df_1d = None

        options = {'maxiter': self.max_iterations}

        if df_1d is None:
            res = scipy.optimize.minimize(f, x0=x0[0, :], method='trust-constr', bounds=self.bounds, jac='2-point',
                                          options=options, constraints=self.constraints, hess=scipy.optimize.BFGS())
        else:
            res = scipy.optimize.minimize(f, x0=x0[0, :], method='trust-constr', bounds=self.bounds, jac=df_1d,
                                          options=options, constraints=self.constraints, hess=scipy.optimize.BFGS())

        result_x = np.atleast_2d(res.x)
        result_fx = np.atleast_2d(res.fun)
        return result_x, result_fx


def _get_scipy_constraints(constraint_list: List[IConstraint]) -> List:
    """
    Converts list of emukit constraint objects to list of scipy constraint objects

    :param constraint_list: List of Emukit constraint objects
    :return: List of scipy constraint objects
    """

    scipy_constraints = []
    for constraint in constraint_list:
        if isinstance(constraint, NonlinearInequalityConstraint):
            if constraint.jacobian_fun is None:
                # No jacobian supplied -> tell scipy to use finite difference method
                jacobian = '2-point'
            else:
                # Jacobian is supplied -> tell scipy to use it
                jacobian = constraint.jacobian_fun

            scipy_constraints.append(
                scipy.optimize.NonlinearConstraint(constraint.fun, constraint.lower_bound, constraint.upper_bound,
                                                   jacobian))
        elif isinstance(constraint, LinearInequalityConstraint):
            scipy_constraints.append(scipy.optimize.LinearConstraint(constraint.constraint_matrix,
                                                                     constraint.lower_bound,
                                                                     constraint.upper_bound))
        else:
            raise ValueError('Constraint type {} not recognised'.format(type(constraint)))
    return scipy_constraints
