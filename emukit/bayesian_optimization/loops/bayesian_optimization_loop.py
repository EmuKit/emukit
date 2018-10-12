# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ...core.loop import OuterLoop, Sequential, FixedIntervalUpdater, LoopState, \
    UserFunctionResult, ModelUpdater, CandidatePointCalculator
from ...core.optimization import AcquisitionOptimizer
from ...core.parameter_space import ParameterSpace
from ...core.acquisition import Acquisition
from ...core.interfaces import IModel

from ..acquisitions.expected_improvement import ExpectedImprovement


class BayesianOptimizationLoop(OuterLoop):
    def __init__(self, model: IModel, space: ParameterSpace, X_init: np.array, Y_init: np.array,
                 acquisition: Acquisition = None, candidate_point_calculator: CandidatePointCalculator = None,
                 model_updater: ModelUpdater = None):

        """
        Emukit class that implement a loop for building modular Bayesian optimization

        :param model: The model that approximates the underlying function
        :param space: Input space where the optimization is carried out.
        :param acquisition: The acquisition function that will be used to collect new points (default, EI).
        :param X_init: x values for initial function evaluations
        :param Y_init: y values for initial function evaluations
        :param model_updater: Defines how and how often the model will be updated if new data
        arrives (default, FixedIntervalUpdater)
        :param candidate_point_calculator: Optimizes the acquisition function to find the
        next candidate to evaluate (default, Sequential)
        """

        if acquisition is None:
            acquisition = ExpectedImprovement(model)

        if model_updater is None:
            model_updater = FixedIntervalUpdater(model, 1)

        if candidate_point_calculator is None:
            acquisition_optimizer = AcquisitionOptimizer(space)
            candidate_point_calculator = Sequential(acquisition, acquisition_optimizer)

        initial_results = []
        for i in range(X_init.shape[0]):
            initial_results.append(UserFunctionResult(X_init[i], Y_init[i]))
        loop_state = LoopState(initial_results)

        super(BayesianOptimizationLoop, self).__init__(candidate_point_calculator, model_updater, loop_state)
