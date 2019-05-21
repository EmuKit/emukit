# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from emukit.bayesian_optimization.loops.bayesian_optimization_loop import BayesianOptimizationResults
from emukit.core.loop.outer_loop import OuterLoop
from emukit.core.loop.candidate_point_calculators import CandidatePointCalculator
from emukit.core.loop.loop_state import LoopState
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.parameter_space import ParameterSpace
from emukit.core.loop.model_updaters import ModelUpdater
from emukit.core.optimization.context_manager import ContextManager


class RandomSampling(CandidatePointCalculator):

    def __init__(self, parameter_space: ParameterSpace):
        """
        Samples a new candidate point uniformly at random

        :param parameter_space: Input space
        """
        self.parameter_space = parameter_space

    def compute_next_points(self, loop_state: LoopState, context_manager: ContextManager) -> np.ndarray:
        """
        :param loop_state: Object that contains current state of the loop
        :param context_manager: Optimization context manager.

        :return: (1 x n_dims) array of next inputs to evaluate the function at
        """

        if context_manager is not None:
            sample = context_manager.context_free_space.sample_uniform(1)
            sample = context_manager.expand_vector(sample)
        else:
            sample = self.parameter_space.sample_uniform(1)
        return sample


class DummyModelUpdate(ModelUpdater):
    def update(self, loop_state: LoopState) -> None:
        """
        Dummy model for random search

        :param loop_state: Object that contains current state of the loop
        """
        pass


class RandomSearch(OuterLoop):
    def __init__(self, space: ParameterSpace, x_init: np.ndarray = None,
                 y_init: np.ndarray = None, cost_init: np.ndarray = None):

        """
        Simple loop to perform random search where in each iteration points are sampled uniformly at random
        over the input space.

        :param space: Input space where the optimization is carried out.
        :param x_init: 2d numpy array of shape (no. points x no. input features) of initial X data
        :param y_init: 2d numpy array of shape (no. points x no. targets) of initial Y data
        :param cost_init: 2d numpy array of shape (no. points x no. targets) of initial cost of each function evaluation
        """

        model_updaters = DummyModelUpdate()

        candidate_point_calculator = RandomSampling(parameter_space=space)

        if x_init is not None and y_init is not None:
            loop_state = create_loop_state(x_init, y_init, cost_init)
        else:
            loop_state = None

        super().__init__(candidate_point_calculator, model_updaters, loop_state=loop_state)

    def get_results(self):
        return BayesianOptimizationResults(self.loop_state)
