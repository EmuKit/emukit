# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import abc

import numpy as np

from .loop_state import LoopState
from .. import ParameterSpace, InformationSourceParameter
from ..acquisition import Acquisition
from ..interfaces import IModel
from ..optimization import AcquisitionOptimizer


class CandidatePointCalculator(abc.ABC):
    """ Computes the next point(s) for function evaluation """
    @abc.abstractmethod
    def compute_next_points(self, loop_state: LoopState, context: dict=None) -> np.ndarray:
        """
        :param loop_state: Object that contains current state of the loop
        :param context: Contains variables to fix through optimization of acquisition function. The dictionary key is
                        the parameter name and the value is the value to fix the parameter to.
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

    def compute_next_points(self, loop_state: LoopState, context: dict=None) -> np.ndarray:
        """
        Computes point(s) to evaluate next

        :param loop_state: Object that contains current state of the loop
        :param context: Contains variables to fix through optimization of acquisition function. The dictionary key is
                        the parameter name and the value is the value to fix the parameter to.
        :return: List of function inputs to evaluate the function at next
        """
        x, _ = self.acquisition_optimizer.optimize(self.acquisition, context)
        return np.atleast_2d(x)


class GreedyBatchPointCalculator(CandidatePointCalculator):
    """
    Batch point calculator. This point calculator calculates the first point in the batch then adds this as a fake
    observation in the model with a Y value equal to the mean prediction. The model is reset with the original data at
    the end of collecting a batch but if you use a model where training the model with the same data leads to different
    predictions, the model behaviour will be modified.
    """
    def __init__(self, model: IModel, acquisition: Acquisition, acquisition_optimizer: AcquisitionOptimizer,
                 batch_size: int=1):
        """
        :param model: Model that is used by the acquisition function
        :param acquisition: Acquisition to be optimized to find each point in batch
        :param acquisition_optimizer: Acquisition optimizer that optimizes acquisition function to find each point in batch
        :param batch_size: Number of points to calculate in batch
        """
        if (not isinstance(batch_size, int)) or (batch_size < 1):
            raise ValueError('Batch size should be a positive integer')
        self.model = model
        self.acquisition = acquisition
        self.acquisition_optimizer = acquisition_optimizer
        self.batch_size = batch_size

    def compute_next_points(self, loop_state: LoopState, context: dict=None) -> np.ndarray:
        """
        :param loop_state: Object containing history of the loop
        :param context: Contains variables to fix through optimization of acquisition function. The dictionary key is
                        the parameter name and the value is the value to fix the parameter to.
        :return: 2d array of size (batch_size x input dimensions) of new points to evaluate
        """
        new_xs = []
        original_data = (self.model.X, self.model.Y)
        for _ in range(self.batch_size):
            new_x, _ = self.acquisition_optimizer.optimize(self.acquisition, context)
            new_xs.append(new_x)
            new_y = self.model.predict(new_x)[0]

            # Add new point as fake observation in model
            all_x = np.concatenate([self.model.X, new_x], axis=0)
            all_y = np.concatenate([self.model.Y, new_y], axis=0)
            self.model.update_data(all_x, all_y)

        # Reset data
        self.model.update_data(*original_data)
        return np.concatenate(new_xs, axis=0)


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

    def compute_next_points(self, loop_state: LoopState=None, context: dict=None) -> np.ndarray:
        """
        Computes the location and source of the next point to evaluate by finding the optimum input location at each
        information source, then picking the information source where the value of the acquisition at the optimum input
        location is highest.

        :param loop_state: Object that tracks the state of the loop. Currently unused
        :param context: Contains variables to fix through optimization of acquisition function. The dictionary key is
                        the parameter name and the value is the value to fix the parameter to.
        :return: A list of function inputs to evaluate next
        """
        f_mins = np.zeros((len(self.source_parameter.domain)))
        x_opts = []

        if context is None:
            context = dict()
        elif self.source_parameter.name in context:
            # Information source parameter already has a context so just optimise the acquisition at this source
            return self.acquisition_optimizer.optimize(self.acquisition, context)[0]

        # Optimize acquisition for each information source
        for i in range(len(self.source_parameter.domain)):
            # Fix the source using a dictionary, the key is the name of the parameter to fix and the value is the
            # value to which the parameter is fixed
            context[self.source_parameter.name] = self.source_parameter.domain[i]
            x, f_mins[i] = self.acquisition_optimizer.optimize(self.acquisition, context)
            x_opts.append(x)
        best_source = np.argmin(f_mins)
        return x_opts[best_source]

