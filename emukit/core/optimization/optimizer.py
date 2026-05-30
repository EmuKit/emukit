# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, List, Optional, Tuple

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

    def optimize(
        self, x0: np.ndarray, f: Callable, df: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimize f starting from x0.

        :param x0: Initial point for optimization
        :param f: Objective function to minimize (required)
        :param df: Gradient of f (optional; if None, gradient will be approximated)
        :return: Tuple of (location of optimum, value at optimum)
        """
        raise NotImplementedError("The optimize method is not implemented in the parent class.")


class OptLbfgs(Optimizer):
    """
    Wrapper for L-BFGS-B optimizer using true or approximate gradients.
    """

    def __init__(self, bounds, max_iterations=1000):
        super(OptLbfgs, self).__init__(bounds)
        self.max_iterations = max_iterations

    def optimize(
        self, x0: np.ndarray, f: Callable, df: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Minimize f using L-BFGS-B.

        :param x0: Initial point for optimization
        :param f: Objective function to minimize (required)
        :param df: Gradient of f (optional; if None, gradient will be approximated)
        :return: Tuple of (location of optimum, value at optimum)
        """

        if df is not None:
            # Use provided gradient
            def f_and_grad(x):
                x_2d = np.atleast_2d(x)
                f_val_raw = f(x_2d)
                # Extract scalar value, handling 0-d arrays
                if np.isscalar(f_val_raw):
                    f_val = float(f_val_raw)
                else:
                    f_arr = np.asarray(f_val_raw).squeeze()
                    f_val = float(f_arr) if f_arr.ndim == 0 else float(f_arr.flat[0])
                
                grad = df(x_2d)
                # Handle gradient shape - could be (n_dims, n_samples) or (n_samples, n_dims)
                if grad.shape[0] == 1:
                    grad_val = grad[0]  # Shape (n_dims,)
                elif grad.shape[1] == 1:
                    grad_val = grad[:, 0]  # Shape (n_dims,) from transpose 
                else:
                    grad_val = grad[0]  # Assume (n_samples, n_dims), take first
                return f_val, grad_val
            
            res = scipy.optimize.fmin_l_bfgs_b(
                f_and_grad, x0=x0, bounds=self.bounds, maxiter=self.max_iterations
            )
        else:
            # Approximate gradient using finite differences
            def f_wrapped(x):
                x_2d = np.atleast_2d(x)
                f_val_raw = f(x_2d)
                # Extract scalar value, handling 0-d arrays
                if np.isscalar(f_val_raw):
                    return float(f_val_raw)
                else:
                    f_arr = np.asarray(f_val_raw).squeeze()
                    return float(f_arr) if f_arr.ndim == 0 else float(f_arr.flat[0])
            
            res = scipy.optimize.fmin_l_bfgs_b(
                f_wrapped, x0=x0, bounds=self.bounds, approx_grad=True, maxiter=self.max_iterations
            )

        # Handle abnormal termination
        if res[2]["task"] == b"ABNORMAL_TERMINATION_IN_LNSRCH":
            result_x = np.atleast_2d(x0)
            result_fx = np.atleast_2d(f(x0))
        else:
            result_x = np.atleast_2d(res[0])
            result_fx = np.atleast_2d(res[1])

        return result_x, result_fx


def apply_optimizer(
    optimizer: Optimizer,
    x0: np.ndarray,
    space: ParameterSpace,
    f: Callable,
    df: Optional[Callable] = None,
    context_manager: ContextManager = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimize f using the provided optimizer, handling context variables.

    :param optimizer: The optimizer to use
    :param x0: Initial point (with or without context variables)
    :param space: Parameter space describing input domain
    :param f: Objective function to minimize (required)
    :param df: Gradient of f (optional; if None, will be approximated)
    :param context_manager: If provided, handles fixed context variables
    :return: Tuple of (optimized point with context, function value at optimum)
    """

    if context_manager is None:
        context_manager = ContextManager(space, {})

    # Build objective functions that handle context
    problem = OptimizationWithContext(x0=x0, f=f, df=df, context_manager=context_manager)

    add_context = lambda x: context_manager.expand_vector(x)

    # Optimize
    optimized_x, _ = optimizer.optimize(
        problem.x0_no_context, 
        problem.f_no_context, 
        problem.df_no_context
    )

    # Add context and round according to parameter types
    suggested_x_with_context = add_context(optimized_x)
    suggested_x_with_context_rounded = space.round(suggested_x_with_context)

    # Evaluate at final point
    f_opt = f(suggested_x_with_context_rounded)
    
    return suggested_x_with_context_rounded, f_opt


class OptimizationWithContext(object):
    """
    Wraps an objective function to handle fixed context variables during optimization.
    """

    def __init__(
        self,
        x0: np.ndarray,
        f: Callable,
        df: Optional[Callable] = None,
        context_manager: ContextManager = None,
    ):
        """
        Constructor of an objective function that takes as input a vector x of the non context variables
        and returns a value in which the context variables have been fixed.
        
        :param x0: Initial point
        :param f: Objective function
        :param df: Gradient of objective function
        :param context_manager: Handles fixed context variables
        """
        self.x0 = np.atleast_2d(x0)
        self.f = f
        self.df = df
        self.context_manager = context_manager

        # Check if context is actually empty (no variables are fixed)
        has_context = context_manager and len(context_manager.context_values) > 0
        
        if not has_context:
            self.x0_no_context = x0
            self.f_no_context = self.f
            # When there's no actual context, use finite differences (scipy converges better)
            # This preserves the behavior of the original code
            self.df_no_context = None
        else:
            self.x0_no_context = self.x0[:, self.context_manager.non_context_idxs]
            self.f_no_context = self._make_f_no_context()
            self.df_no_context = self._make_df_no_context() if df is not None else None

    def _make_f_no_context(self) -> Callable:
        """Create wrapper that adds context variables before calling f."""
        def f_no_context(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            xx = self.context_manager.expand_vector(x)
            if x.shape[0] == 1:
                return self.f(xx)[0]
            else:
                return self.f(xx)
        return f_no_context

    def _make_df_no_context(self) -> Callable:
        """Create wrapper that extracts gradient for non-context variables."""
        def df_no_context(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(x)
            xx = self.context_manager.expand_vector(x)
            df_xx = self.df(xx)
            # Handle gradient shape - could be (n_dims, n_samples) or (n_samples, n_dims)
            # Extract only non-context dimensions
            if df_xx.shape[0] == 1 or len(df_xx.shape) == 1:
                # Shape (n_dims,) or (1, n_dims) - transpose handling
                return df_xx[:, np.array(self.context_manager.non_context_idxs)] if len(df_xx.shape) > 1 else df_xx[np.array(self.context_manager.non_context_idxs)]
            else:
                # Shape (n_dims, n_samples) - extract rows for non-context dims
                return df_xx[np.array(self.context_manager.non_context_idxs), :]
        return df_no_context


class OptTrustRegionConstrained(Optimizer):
    """
    Wrapper for Trust-Region Constrained algorithm that can deal with non-linear constraints
    """

    def __init__(self, bounds: List[Tuple], constraints: List[IConstraint], max_iterations: int = 1000):
        """
        :param bounds: List of tuples containing (lower_bound, upper_bound) for each parameter
        :param constraints: List of constraints, can contain a mix of linear and non-linear constraints
        :param max_iterations: Maximum number of iterations before the optimizer is stopped
        """
        super().__init__(bounds)
        self.max_iterations = max_iterations
        self.constraints = _get_scipy_constraints(constraints)

    def optimize(
        self, x0: np.ndarray, f: Callable, df: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Trust region constrained optimization algorithm

        :param x0: Initial start point
        :param f: Objective function to minimize (required)
        :param df: Derivative of function to optimize (optional)
        :return: Location of optimum and function value at optimum
        """

        # Prepare gradient function
        if df is not None:
            # Convert 2d output to 1d for scipy, handle 1d scipy input
            def df_1d(x):
                x_2d = np.atleast_2d(x)
                grad = df(x_2d)
                # Handle gradient shape - could be (n_dims, n_samples) or (n_samples, n_dims)
                if grad.ndim == 1:
                    return grad  # Already 1D
                elif grad.shape[0] == 1:
                    return grad[0]  # Extract from (1, n_dims)
                elif grad.shape[1] == 1:
                    return grad[:, 0]  # Extract from (n_dims, 1) (transposed)
                else:
                    return grad[0, :]  # Assume (n_samples, n_dims), take first
        else:
            # Let scipy approximate with finite differences
            df_1d = "2-point"
        
        # Wrap f to handle 1d scipy input
        def f_wrapped(x):
            x_2d = np.atleast_2d(x)
            f_val = f(x_2d)
            # Extract scalar value, handling both scalar and array returns (including 0-d)
            if np.isscalar(f_val):
                return float(f_val)
            else:
                f_arr = np.asarray(f_val).squeeze()
                return float(f_arr) if f_arr.ndim == 0 else float(f_arr.flat[0] if f_arr.size > 0 else 0.0)

        options = {"maxiter": self.max_iterations}

        # Handle both 1D and 2D x0
        x0_1d = np.atleast_1d(x0).flatten()
        
        res = scipy.optimize.minimize(
            f_wrapped,
            x0=x0_1d,
            method="trust-constr",
            bounds=self.bounds,
            jac=df_1d,
            options=options,
            constraints=self.constraints,
            hess=scipy.optimize.BFGS(),
        )

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
                jacobian = "2-point"
            else:
                # Jacobian is supplied -> tell scipy to use it
                jacobian = constraint.jacobian_fun

            scipy_constraints.append(
                scipy.optimize.NonlinearConstraint(
                    constraint.fun, constraint.lower_bound, constraint.upper_bound, jacobian
                )
            )
        elif isinstance(constraint, LinearInequalityConstraint):
            scipy_constraints.append(
                scipy.optimize.LinearConstraint(
                    constraint.constraint_matrix, constraint.lower_bound, constraint.upper_bound
                )
            )
        else:
            raise ValueError("Constraint type {} not recognised".format(type(constraint)))
    return scipy_constraints
