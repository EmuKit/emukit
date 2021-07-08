# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import Optional, Tuple

import numpy as np

from .. import ParameterSpace
from ..acquisition import Acquisition
from .context_manager import Context, ContextManager


class AcquisitionOptimizerBase(abc.ABC):
    """
    Base class for acquisition optimizers
    """
    def __init__(self, space: ParameterSpace):
        """
        :param space: Parameter space containing entire input domain including any context variables
        """
        self.space = space

    @abc.abstractmethod
    def _optimize(self, acquisition: Acquisition, context_manager: ContextManager) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of optimization. See class docstring for details.

        :param acquisition: The acquisition function to be optimized
        :param context_manager: Optimization context manager.
        :return: Tuple of (location of maximum, acquisition value at maximizer)
        """
        pass

    def optimize(self, acquisition: Acquisition, context: Optional[Context] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimizes the acquisition function.

        :param acquisition: The acquisition function to be optimized
        :param context: Optimization context.
                        Determines whether any variable values should be fixed during the optimization
        :return: Tuple of (location of maximum, acquisition value at maximizer)
        """
        if context is None:
            context = dict()
        context_manager = ContextManager(self.space, context)
        max_x, max_value = self._optimize(acquisition, context_manager)

        # Optimization might not match any encoding exactly
        # Rounding operation here finds the closest encoding
        rounded_max_x = self.space.round(max_x)

        if not np.array_equal(max_x, rounded_max_x):
            # re-evaluate if x changed while rounding to make sure value is correct
            return rounded_max_x, acquisition.evaluate(rounded_max_x)
        else:
            return max_x, max_value
