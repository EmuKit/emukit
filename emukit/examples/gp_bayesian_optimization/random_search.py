# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.core.loop.outer_loop import OuterLoop
from emukit.bayesian_optimization.loops.bayesian_optimization_loop import BayesianOptimizationResults
from emukit.core.loop.candidate_point_calculators import CandidatePointCalculator
from emukit.experimental_design.model_free.random_design import RandomDesign
from emukit.core.loop.loop_state import LoopState
from emukit.core.parameter_space import ParameterSpace
from emukit.core.loop.model_updaters import ModelUpdater
from emukit.core.loop.loop_state import create_loop_state


class RandomSampling(CandidatePointCalculator):

    def __init__(self, parameter_space: ParameterSpace):
        """
        Samples a new candidate point uniformly at random

        :param parameter_space: Input space
        """
        self.rd = RandomDesign(parameter_space=parameter_space)

    def compute_next_points(self, loop_state: LoopState, context: dict=None) -> np.ndarray:
        """
        :param loop_state: Object that contains current state of the loop
        :param context: Contains variables to fix through optimization of acquisition function. The dictionary key is
                        the parameter name and the value is the value to fix the parameter to.
        :return: (n_points x n_dims) array of next inputs to evaluate the function at
        """
        return self.rd.get_samples(1)


class DummyModelUpdate(ModelUpdater):
    def update(self, loop_state: LoopState) -> None:
        """
        Dummy model for random search

        :param loop_state: Object that contains current state of the loop
        """
        pass


class RandomSearch(OuterLoop):
    def __init__(self, space: ParameterSpace, x_init: np.ndarray=None,
                 y_init: np.ndarray=None, cost_init: np.ndarray=None):

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

