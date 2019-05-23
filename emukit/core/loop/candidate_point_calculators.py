# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import abc

import numpy as np

from .loop_state import LoopState
from ..acquisition import Acquisition
from ..interfaces import IModel
from ..optimization.acquisition_optimizer import AcquisitionOptimizerBase
from ..optimization.context_manager import ContextManager
from ..parameter_space import ParameterSpace


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


class SequentialPointCalculator(CandidatePointCalculator):
    """ This candidate point calculator chooses one candidate point at a time """
    def __init__(self, acquisition: Acquisition, acquisition_optimizer: AcquisitionOptimizerBase) -> None:
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
        self.acquisition.update_parameters()
        x, _ = self.acquisition_optimizer.optimize(self.acquisition, context)
        return x


class GreedyBatchPointCalculator(CandidatePointCalculator):
    """
    Batch point calculator. This point calculator calculates the first point in the batch then adds this as a fake
    observation in the model with a Y value equal to the mean prediction. The model is reset with the original data at
    the end of collecting a batch but if you use a model where training the model with the same data leads to different
    predictions, the model behaviour will be modified.
    """
    def __init__(self, model: IModel, acquisition: Acquisition, acquisition_optimizer: AcquisitionOptimizerBase,
                 batch_size: int=1):
        """
        :param model: Model that is used by the acquisition function
        :param acquisition: Acquisition to be optimized to find each point in batch
        :param acquisition_optimizer: Acquisition optimizer that optimizes acquisition function
                                      to find each point in batch
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
        self.acquisition.update_parameters()
        new_xs = []
        original_data = (self.model.X, self.model.Y)
        for _ in range(self.batch_size):
            new_x, _ = self.acquisition_optimizer.optimize(self.acquisition, context)
            new_xs.append(new_x)
            new_y = self.model.predict(new_x)[0]

            # Add new point as fake observation in model
            all_x = np.concatenate([self.model.X, new_x], axis=0)
            all_y = np.concatenate([self.model.Y, new_y], axis=0)
            self.model.set_data(all_x, all_y)

        # Reset data
        self.model.set_data(*original_data)
        return np.concatenate(new_xs, axis=0)


class RandomSampling(CandidatePointCalculator):
    """
    Samples a new candidate point uniformly at random

    """
    def __init__(self, parameter_space: ParameterSpace):
        """
        :param parameter_space: Input space
        """
        self.parameter_space = parameter_space

    def compute_next_points(self, loop_state: LoopState, context: dict=None) -> np.ndarray:
        """
        :param loop_state: Object that contains current state of the loop
        :param context: Contains variables to fix through optimization of acquisition function. The dictionary key is
                        the parameter name and the value is the value to fix the parameter to.
        :return: (1 x n_dims) array of next inputs to evaluate the function at
        """

        if context is not None:
            context_manager = ContextManager(self.parameter_space, context)
            sample = context_manager.contextfree_space.sample_uniform(1)
            sample = context_manager.expand_vector(sample)
        else:
            sample = self.parameter_space.sample_uniform(1)
        return sample
