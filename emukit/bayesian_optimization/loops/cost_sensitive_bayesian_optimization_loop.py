# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ...bayesian_optimization.acquisitions import ExpectedImprovement
from ...core.acquisition import Acquisition, acquisition_per_expected_cost
from ...core.interfaces import IModel
from ...core.loop import FixedIntervalUpdater, OuterLoop, SequentialPointCalculator
from ...core.loop.loop_state import create_loop_state
from ...core.optimization import AcquisitionOptimizer
from ...core.parameter_space import ParameterSpace


class CostSensitiveBayesianOptimizationLoop(OuterLoop):
    def __init__(self, space: ParameterSpace, model_objective: IModel, model_cost: IModel,
                 acquisition: Acquisition = None, update_interval: int = 1):

        """
        Emukit class that implements a loop for building modular cost sensitive Bayesian optimization.

        :param space: Input space where the optimization is carried out.
        :param model_objective: The model that approximates the underlying objective function
        :param model_cost: The model that approximates the cost of evaluating the objective function
        :param acquisition: The acquisition function that will be used to collect new points (default, EI).
        :param update_interval:  Number of iterations between optimization of model hyper-parameters. Defaults to 1.
        """

        if not np.all(np.isclose(model_objective.X, model_cost.X)):
            raise ValueError('Emukit currently only supports identical '
                             'training inputs for the cost and objective model')

        if acquisition is None:
            expected_improvement = ExpectedImprovement(model_objective)
            acquisition = acquisition_per_expected_cost(expected_improvement, model_cost)

        model_updater_objective = FixedIntervalUpdater(model_objective, update_interval)
        model_updater_cost = FixedIntervalUpdater(model_cost, update_interval, lambda state: state.cost)

        acquisition_optimizer = AcquisitionOptimizer(space)
        candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)

        loop_state = create_loop_state(model_objective.X, model_objective.Y, model_cost.Y)

        super(CostSensitiveBayesianOptimizationLoop, self).__init__(candidate_point_calculator,
                                                                    [model_updater_objective, model_updater_cost],
                                                                    loop_state)
