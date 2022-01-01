# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple

import numpy as np

from .. import ParameterSpace
from ..acquisition import Acquisition
from .acquisition_optimizer import AcquisitionOptimizerBase
from .anchor_points_generator import ObjectiveAnchorPointsGenerator
from .context_manager import ContextManager
from .optimizer import OptLbfgs, OptTrustRegionConstrained, apply_optimizer

_log = logging.getLogger(__name__)


class GradientAcquisitionOptimizer(AcquisitionOptimizerBase):
    """ Optimizes the acquisition function using a quasi-Newton method (L-BFGS).
    Can be used for continuous acquisition functions.
    """
    def __init__(self, space: ParameterSpace, num_samples=1000, num_anchor=1) -> None:
        """
        :param space: The parameter space spanning the search problem.
        """
        super().__init__(space)
        self.num_samples = num_samples
        self.num_anchor = num_anchor

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

        optimizer = self._get_optimizer(context_manager)
        anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, acquisition, num_samples=self.num_samples)

        # Select the anchor points (with context)
        anchor_points = anchor_points_generator.get(num_anchor=self.num_anchor, context_manager=context_manager)

        _log.info("Starting gradient-based optimization of acquisition function {}".format(type(acquisition)))
        optimized_points = []
        for a in anchor_points:
            optimized_point = apply_optimizer(optimizer, a, space=self.space, f=f, df=None, f_df=f_df,
                                              context_manager=context_manager)
            optimized_points.append(optimized_point)

        x_min, fx_min = min(optimized_points, key=lambda t: t[1])
        return x_min, -fx_min

    def _get_optimizer(self, context_manager):
        if len(self.space.constraints) == 0:
            return OptLbfgs(context_manager.contextfree_space.get_bounds())
        else:
            return OptTrustRegionConstrained(context_manager.contextfree_space.get_bounds(), self.space.constraints)
