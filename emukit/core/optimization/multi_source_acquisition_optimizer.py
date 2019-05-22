# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Optional

import numpy as np

from .. import InformationSourceParameter, ParameterSpace
from ..acquisition import Acquisition
from .acquisition_optimizer import AcquisitionOptimizerBase
from .context_manager import ContextManager, Context


class MultiSourceAcquisitionOptimizer(AcquisitionOptimizerBase):
    """
    Optimizes the acquisition function by finding the optimum input location at each information source, then picking
    the information source where the value of the acquisition at the optimum input location is highest.
    """
    def __init__(self, acquisition_optimizer: AcquisitionOptimizerBase, space: ParameterSpace) -> None:
        """
        :param acquisition_optimizer: Optimizer to use for optimizing the acquisition once the information source
                                      has been fixed
        :param space: Domain to search for maximum over
        """
        super().__init__(space)
        self.acquisition_optimizer = acquisition_optimizer
        self.source_parameter = self._get_information_source_parameter()
        self.n_sources = np.array(self.source_parameter.domain).size

    def _get_information_source_parameter(self) -> InformationSourceParameter:
        """
        :return: The parameter containing the index of the information source
        """
        source_parameter = [param for param in self.space.parameters if isinstance(param, InformationSourceParameter)]
        if len(source_parameter) == 0:
            raise ValueError('No source parameter found')
        return source_parameter[0]

    def _optimize(self, acquisition: Acquisition, context_manager: ContextManager) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of abstract method. Does nothing, optimize overwritten instead.

        See AcquisitionOptimizerBase._optimizer for parameter descriptions.
        See class docstring for implementation details.
        """
        pass

    def optimize(self, acquisition: Acquisition, context: Optional[Context] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the location and source of the next point to evaluate by finding the maximum input location at each
        information source, then picking the information source where the value of the acquisition at the optimum input
        location is highest.

        :param acquisition: The acquisition function to be optimized
        :param context: Contains variables to fix through optimization of acquisition function. The dictionary key is
                        the parameter name and the value is the value to fix the parameter to.
        :return: A tuple of (location of maximum, acquisition value at maximum)
        """
        f_maxs = np.zeros((len(self.source_parameter.domain)))
        x_opts = []

        if context is None:
            context = dict()
        elif self.source_parameter.name in context:
            # Information source parameter already has a context so just optimise the acquisition at this source
            return self.acquisition_optimizer.optimize(acquisition, context)

        # Optimize acquisition for each information source
        for i in range(len(self.source_parameter.domain)):
            # Fix the source using a dictionary, the key is the name of the parameter to fix and the value is the
            # value to which the parameter is fixed
            context[self.source_parameter.name] = self.source_parameter.domain[i]
            x, f_maxs[i] = self.acquisition_optimizer.optimize(acquisition, context)
            x_opts.append(x)
        best_source = np.argmax(f_maxs)
        return x_opts[best_source], np.max(f_maxs)
