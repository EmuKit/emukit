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


class RandomSampling(CandidatePointCalculator):

    def __init__(self, parameter_space):
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
        pass


class RandomSearch(OuterLoop):
    def __init__(self, space: ParameterSpace):

        """
        Emukit class that implement a loop for random search

        :param space: Input space where the optimization is carried out.
        """

        model_updaters = DummyModelUpdate()

        candidate_point_calculator = RandomSampling(parameter_space=space)

        super().__init__(candidate_point_calculator, model_updaters, loop_state=None)

    def get_results(self):
        return BayesianOptimizationResults(self.loop_state)

