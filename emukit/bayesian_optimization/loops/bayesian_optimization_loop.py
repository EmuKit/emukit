# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ...core.acquisition import Acquisition
from ...core.interfaces import IDifferentiable, IModel
from ...core.loop import FixedIntervalUpdater, OuterLoop, SequentialPointCalculator
from ...core.loop.loop_state import create_loop_state, LoopState
from ...core.optimization import AcquisitionOptimizerBase
from ...core.optimization import GradientAcquisitionOptimizer
from ...core.parameter_space import ParameterSpace
from ..acquisitions import ExpectedImprovement
from ..acquisitions.log_acquisition import LogAcquisition
from ..local_penalization_calculator import LocalPenalizationPointCalculator


import logging
_log = logging.getLogger(__name__)

class BayesianOptimizationLoop(OuterLoop):
    def __init__(self, space: ParameterSpace, model: IModel, acquisition: Acquisition = None, update_interval: int = 1,
                 batch_size: int = 1, acquisition_optimizer: AcquisitionOptimizerBase = None):

        """
        Emukit class that implement a loop for building modular Bayesian optimization

        :param space: Input space where the optimization is carried out.
        :param model: The model that approximates the underlying function
        :param acquisition: The acquisition function that will be used to collect new points (default, EI). If batch
                            size is greater than one, this acquisition must output positive values only.
        :param update_interval: Number of iterations between optimization of model hyper-parameters. Defaults to 1.
        :param batch_size: How many points to evaluate in one iteration of the optimization loop. Defaults to 1.
        :param acquisition_optimizer: Optimizer selecting next evaluation points
                                      by maximizing acquisition.
                                      Gradient based optimizer is used if None.
                                      Defaults to None.
        """

        self.model = model

        if acquisition is None:
            acquisition = ExpectedImprovement(model)

        model_updaters = FixedIntervalUpdater(model, update_interval)

        if acquisition_optimizer is None:
            acquisition_optimizer = GradientAcquisitionOptimizer(space)
        if batch_size == 1:
            _log.info("Batch size is 1, using SequentialPointCalculator")
            candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)
        else:
            _log.info("Batch size is " + str(batch_size) + ", using LocalPenalizationPointCalculator")
            log_acquisition = LogAcquisition(acquisition)
            candidate_point_calculator = LocalPenalizationPointCalculator(log_acquisition, acquisition_optimizer, model,
                                                                          space, batch_size)

        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updaters, loop_state)

    def get_results(self):
        return BayesianOptimizationResults(self.loop_state)


class BayesianOptimizationResults:
    def __init__(self, loop_state: LoopState):

        """
        Emukit class that takes as input the loop state and computes some results.

        :param loop_state: The loop state it its current form. Currently it only contains X and Y.
        """

        self.minimum_location = loop_state.X[np.argmin(loop_state.Y), :]
        self.minimum_value = np.min(loop_state.Y)
        self.best_found_value_per_iteration = np.minimum.accumulate(loop_state.Y).flatten()
