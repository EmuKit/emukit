# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from emukit.bayesian_optimization.loops.cost_sensitive_bayesian_optimization_loop import \
    CostSensitiveBayesianOptimizationLoop
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core import InformationSourceParameter
from emukit.examples.fabolas.model import FabolasModel

from ...core.acquisition import acquisition_per_expected_cost
from ...core.loop import FixedIntervalUpdater, SequentialPointCalculator
from ...core.loop.loop_state import create_loop_state
from ...core.optimization import AcquisitionOptimizerBase
from ...core.optimization import RandomSearchAcquisitionOptimizer
from .continuous_fidelity_entropy_search import ContinuousFidelityEntropySearch
from emukit.core.acquisition import IntegratedHyperParameterAcquisition


class FabolasLoop(CostSensitiveBayesianOptimizationLoop):

    def __init__(self, space: ParameterSpace,
                 X_init: np.ndarray, Y_init: np.ndarray, cost_init: np.ndarray,
                 s_min: float, s_max: float,
                 update_interval: int = 1,
                 num_eval_points: int = 500,
                 acquisition_optimizer: AcquisitionOptimizerBase = None):
        """
        Emukit class that implements a loop for building modular cost sensitive Bayesian optimization.

        :param space: Input space where the optimization is carried out.
        :param model_objective: The model that approximates the underlying objective function
        :param model_cost: The model that approximates the cost of evaluating the objective function
        :param acquisition: The acquisition function that will be used to collect new points (default, EI).
        :param update_interval:  Number of iterations between optimization of model hyper-parameters. Defaults to 1.
        :param acquisition_optimizer: Optimizer selecting next evaluation points
                                      by maximizing acquisition.
                                      Gradient based optimizer is used if None.
                                      Defaults to None.
        """

        l = space.parameters
        l.extend([ContinuousParameter("dataset_size",s_min, s_max)])
        extended_space = ParameterSpace(l)

        model_objective = FabolasModel(X_init=X_init, Y_init=Y_init, s_min=s_min, s_max=s_max)
        model_cost = FabolasModel(X_init=X_init, Y_init=cost_init[:, None], s_min=s_min, s_max=s_max)

        # entropy_search = ContinuousFidelityEntropySearch(model_objective, space=extended_space,
        #                                                      target_fidelity_index=len(extended_space.parameters) - 1)

        acquisition_generator = lambda model: ContinuousFidelityEntropySearch(model_objective, space=extended_space,
                                                                              target_fidelity_index=len(extended_space.parameters) - 1)
        entropy_search = IntegratedHyperParameterAcquisition(model_objective, acquisition_generator)

        acquisition = acquisition_per_expected_cost(entropy_search, model_cost)

        model_updater_objective = FixedIntervalUpdater(model_objective, update_interval)
        model_updater_cost = FixedIntervalUpdater(model_cost, update_interval, lambda state: state.cost)

        acquisition_optimizer = RandomSearchAcquisitionOptimizer(extended_space, num_eval_points=num_eval_points)
        candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)

        loop_state = create_loop_state(model_objective.X, model_objective.Y, model_cost.Y)

        super(CostSensitiveBayesianOptimizationLoop, self).__init__(candidate_point_calculator,
                                                                    [model_updater_objective, model_updater_cost],
                                                                    loop_state)
