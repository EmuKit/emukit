import logging
from typing import Tuple

import numpy as np
from GPyOpt.optimization.acquisition_optimizer import ContextManager

from .. import ParameterSpace
from ..acquisition import Acquisition
from ..optimization.acquisition_optimizer import AcquisitionOptimizerBase

_log = logging.getLogger(__name__)


class RandomSearchAcquisitionOptimizer(AcquisitionOptimizerBase):
    """ Optimizes the acquisition function by evaluating at random points."""
    def __init__(self, space: ParameterSpace, num_samples: int, **kwargs) -> None:
        self.space = space
        self.num_samples = num_samples

    def optimize(self, acquisition: Acquisition, context: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimizes the acquisition function, taking into account gradients if it supports them
        :param acquisition: The acquisition function to be optimized
        :param context: Optimization context.
                        Determines whether any variable values should be fixed during the optimization
        :return: Tuple of (location of maximum, acquisition value at maximizer)
        """
        if context is not None:
            context_manager = ContextManager(
                self.space.convert_to_gpyopt_design_space(), context)
            noncontext_space = ParameterSpace(
                [param for param in self.space.parameters if param.name in context])
        else:
            context_manager = None
            noncontext_space = self.space

        _log.info("Starting random search optimization of acquisition function {}"
                  .format(type(acquisition)))
        samples = noncontext_space.sample_uniform(self.num_samples)
        if context_manager is not None:
            samples = context_manager._expand_vector(samples)
        acquisition_values = acquisition.evaluate(samples)
        max_sample_index = np.argmax(acquisition_values)
        max_sample = samples[[max_sample_index]]

        rounded_max_sample = self.space.round(max_sample)
        rounded_max_value = acquisition.evaluate(rounded_max_sample)
        return rounded_max_sample, rounded_max_value
