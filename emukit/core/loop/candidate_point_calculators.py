# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import abc

import numpy as np

from .loop_state import LoopState
from .. import ParameterSpace, InformationSourceParameter
from ..acquisition import Acquisition
from ..optimization import AcquisitionOptimizer


class CandidatePointCalculator(abc.ABC):
    """ Computes the next point(s) for function evaluation """
    @abc.abstractmethod
    def compute_next_points(self, loop_state: LoopState) -> np.ndarray:
        """
        :param loop_state: Object that contains current state of the loop
        :return: (n_points x n_dims) array of next inputs to evaluate the function at
        """
        pass


class Sequential(CandidatePointCalculator):
    """ This candidate point calculator chooses one candidate point at a time """
    def __init__(self, acquisition: Acquisition, acquisition_optimizer: AcquisitionOptimizer) -> None:
        """
        :param acquisition: Acquisition function to maximise
        :param acquisition_optimizer: Optimizer of acquisition function
        """
        self.acquisition = acquisition
        self.acquisition_optimizer = acquisition_optimizer

    def compute_next_points(self, loop_state: LoopState) -> np.ndarray:
        """
        Computes point(s) to evaluate next

        :param loop_state: Object that contains current state of the loop
        :return: List of function inputs to evaluate the function at next
        """
        x, _ = self.acquisition_optimizer.optimize(self.acquisition)
        return np.atleast_2d(x)


class MultiSourceSequential(CandidatePointCalculator):
    """
    Finds the location and information source of the next point to evaluate
    """
    def __init__(self, acquisition: Acquisition, acquisition_optimizer: AcquisitionOptimizer, space: ParameterSpace) -> None:
        """
        :param acquisition: Acquisition function to find maximum of
        :param acquisition_optimizer: Optimizer of the acquisition function
        :param space: Domain to search for maximum over
        """
        self.acquisition = acquisition
        self.acquisition_optimizer = acquisition_optimizer
        self.space = space
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

    def compute_next_points(self, loop_state: LoopState=None) -> np.ndarray:
        """
        Computes the location and source of the next point to evaluate by finding the optimum input location at each
        information source, then picking the information source where the value of the acquisition at the optimum input
        location is highest.

        :param loop_state: Object that tracks the state of the loop. Currently unused
        :return: A list of function inputs to evaluate next
        """
        f_mins = np.zeros((len(self.source_parameter.domain)))
        x_opts = []

        # Optimize acquisition for each information source
        for i in range(len(self.source_parameter.domain)):
            # Fix the source using a dictionary, the key is the name of the parameter to fix and the value is the
            # value to which the parameter is fixed
            context = {self.source_parameter.name: self.source_parameter.domain[i]}
            x, f_mins[i] = self.acquisition_optimizer.optimize(self.acquisition, context)
            x_opts.append(x)
        best_source = np.argmin(f_mins)
        return x_opts[best_source]
