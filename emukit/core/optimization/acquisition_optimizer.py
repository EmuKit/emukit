# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import GPyOpt
import numpy as np

from .. import ParameterSpace
from ..acquisition import Acquisition


import logging
_log = logging.getLogger(__name__)


class AcquisitionOptimizerBase:
    def optimize(self, acquisition: Acquisition, context: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimizes the acquisition function, taking into account gradients if it supports them

        :param acquisition: The acquisition function to be optimized
        :param context: Optimization context.
                        Determines whether any variable values should be fixed during the optimization
        :return: Tuple of (location of optimum, acquisition value at optimum)
        """
        pass


class AcquisitionOptimizer(AcquisitionOptimizerBase):
    """ Optimizes the acquisition function """
    def __init__(self, space: ParameterSpace, **kwargs) -> None:
        self.space = space
        self.gpyopt_space = space.convert_to_gpyopt_design_space()
        self.gpyopt_acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(self.gpyopt_space, **kwargs)

    def optimize(self, acquisition: Acquisition, context: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimizes the acquisition function, taking into account gradients if it supports them

        :param acquisition: The acquisition function to be optimized
        :param context: Optimization context.
                        Determines whether any variable values should be fixed during the optimization
        :return: Tuple of (location of optimum, acquisition value at optimum)
        """

        # Take negative of acquisition function because they are to be maximised and the optimizers minimise
        f = lambda x: -acquisition.evaluate(x)

        self.gpyopt_acquisition_optimizer.context_manager = GPyOpt.optimization.acquisition_optimizer.ContextManager(
            self.gpyopt_space, context)

        # Context validation
        if context is not None:
            self._validate_context_parameters(context)

            # Return without optimizing if no parameter left to optimize
            are_all_parameters_fixed = len(context.keys()) == len(self.space.parameter_names)
            if are_all_parameters_fixed:
                _log.warning("All parameters are fixed through context")

                x = np.array(self.gpyopt_acquisition_optimizer.context_manager.context_value)[None, :]
                return x, f(x)

        def f_df(x):
            f_value, df_value = acquisition.evaluate_with_gradients(x)
            return -f_value, -df_value

        if acquisition.has_gradients:
            _log.info("Starting gradient-based optimization of acquisition function {}".format(type(acquisition)))
            x, f_min = self.gpyopt_acquisition_optimizer.optimize(f, None, f_df)
        else:
            _log.info("Starting gradient-free optimization of acquisition function {}".format(type(acquisition)))
            x, f_min = self.gpyopt_acquisition_optimizer.optimize(f, None, None)

        # Optimizer sees variables that represent encoded categories as a bunch of continuous values in a range
        # So the output of the optimizer does not necesseraly match any encoding exactly
        # Rounding operation here fins the closes encoding
        rounded_x = self.space.round(x)
        # We may have changed the value of x, so we need to re-evaluate acquisition to make sure f_min is correct
        rounded_f_min = acquisition.evaluate(rounded_x)

        return rounded_x, rounded_f_min

    def _validate_context_parameters(self, context):
        for context_name, context_value in context.items():
            # Check parameter exists in space
            if context_name not in self.space.parameter_names:
                raise ValueError(context_name + ' appears as variable in context but not in the parameter space.')

            # Log warning if context parameter is out of domain
            param = self.space.get_parameter_by_name(context_name)
            if param.check_in_domain(context_value) is False:
                _log.warning(context_name + ' with value ' + str(context_value), ' is out of the domain')
            else:
                _log.info('Parameter ' + context_name + ' fixed to ' + str(context_value))
