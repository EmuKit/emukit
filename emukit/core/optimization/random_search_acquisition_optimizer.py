# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple

import numpy as np

from .acquisition_optimizer import AcquisitionOptimizerBase
from .context_manager import ContextManager
from .. import ParameterSpace
from ..acquisition import Acquisition

_log = logging.getLogger(__name__)


class RandomSearchAcquisitionOptimizer(AcquisitionOptimizerBase):
    """
    Optimizes the acquisition function by evaluating at random points.
    Can be used for discrete and continuous acquisition functions.
    """
    def __init__(self, space: ParameterSpace, num_eval_points: int = 10) -> None:
        """
        :param space: The parameter space spanning the search problem.
        :param num_eval_points: Number of random sampled points which are evaluated per optimization.
        """
        super().__init__(space)
        self.num_eval_points = num_eval_points

    def _optimize(self, acquisition: Acquisition, context_manager: ContextManager)\
        -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of abstract method.

        See AcquisitionOptimizerBase._optimizer for parameter descriptions.
        See class docstring for implementation details.
        """
        _log.info("Starting random search optimization of acquisition function {}"
                  .format(type(acquisition)))
        samples_contextfree = context_manager.contextfree_space.sample_uniform(
            self.num_eval_points)
        samples = context_manager.expand_vector(samples_contextfree)
        acquisition_values = acquisition.evaluate(samples)
        max_index = np.argmax(acquisition_values)

        return samples[[max_index]], acquisition_values[[max_index]]
