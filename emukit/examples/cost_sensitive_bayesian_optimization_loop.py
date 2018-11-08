# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ..bayesian_optimization.acquisitions import ExpectedImprovement
from ..bayesian_optimization.acquisitions.acquisition_per_cost import acquisition_per_expected_cost
from ..core.acquisition import Acquisition
from ..core.interfaces import IModel
from ..core.loop import CandidatePointCalculator, FixedIntervalUpdater, ModelUpdater, OuterLoop, Sequential
from ..core.loop.loop_state import create_loop_state
from ..core.optimization import AcquisitionOptimizer
from ..core.parameter_space import ParameterSpace


class CostSensitiveBayesianOptimizationLoop(OuterLoop):
    def __init__(self, model_objective: IModel, model_cost: IModel,
                 space: ParameterSpace, X_init: np.array, Y_init: np.array, cost_init: np.array,
                 acquisition: Acquisition = None, candidate_point_calculator: CandidatePointCalculator = None,
                 model_updater_objective: ModelUpdater = None, model_updater_cost: ModelUpdater = None):

        """
        Emukit class that implement a loop for building modular cost sensitive Bayesian optimization.

        :param model_objective: The model that approximates the underlying objective function
        :param model_cost: The model that approximates the cost of evaluating the objective function
        :param space: Input space where the optimization is carried out.
        :param acquisition: The acquisition function that will be used to collect new points (default, EI).
        :param X_init: x values for initial function evaluations
        :param Y_init: y values for initial function evaluations
        :param cost_init: costs for initial function evaluations
        :param model_updater_objective: Defines how and how often the model for the objective function
        will be updated if new data arrives (default, FixedIntervalUpdater)
        :param model_updater_cost: Defines how and how often the model for the cost function
        will be updated if new data arrives (default, FixedIntervalUpdater)
        :param candidate_point_calculator: Optimizes the acquisition function to find the
        next candidate to evaluate (default, Sequential)
        """

        if acquisition is None:
            expected_improvement = ExpectedImprovement(model_objective)
            acquisition = acquisition_per_expected_cost(expected_improvement, model_cost)

        if model_updater_objective is None:
            model_updater_objective = FixedIntervalUpdater(model_objective, 1)
        if model_updater_cost is None:
            model_updater_cost = FixedIntervalUpdater(model_cost, 1)

        if candidate_point_calculator is None:
            acquisition_optimizer = AcquisitionOptimizer(space)
            candidate_point_calculator = Sequential(acquisition, acquisition_optimizer)

        loop_state = create_loop_state(X_init, Y_init, cost_init)

        super(CostSensitiveBayesianOptimizationLoop, self).__init__(candidate_point_calculator,
                                                                    [model_updater_objective, model_updater_cost],
                                                                    loop_state)
