# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import numpy as np

from ..acquisitions import ExpectedImprovement
from ..acquisitions.log_acquisition import LogAcquisition
from ..local_penalization_calculator import LocalPenalizationPointCalculator
from ...core.interfaces import IDifferentiable
from ...core.loop.loop_state import create_loop_state, LoopState
from ...core.loop import OuterLoop, SequentialPointCalculator, FixedIntervalUpdater, ModelUpdater
from ...core.optimization import AcquisitionOptimizer
from ...core.parameter_space import ParameterSpace
from ...core.acquisition import Acquisition
from ...core.interfaces import IModel


class BayesianOptimizationLoop(OuterLoop):
    def __init__(self, model: IModel, space: ParameterSpace, acquisition: Acquisition = None, batch_size: int = 1,
                 model_updaters: ModelUpdater = None):

        """
        Emukit class that implement a loop for building modular Bayesian optimization

        :param model: The model that approximates the underlying function
        :param space: Input space where the optimization is carried out.
        :param acquisition: The acquisition function that will be used to collect new points (default, EI). If batch
                            size is greater than one, this acquisition must output positive values only.
        :param batch_size: How many points to evaluate in one iteration of the optimization loop. Defaults to 1.
        :param model_updaters: Defines how and when the model is updated if new data arrives.
                              Defaults to updating hyper-parameters every iteration.
        """

        self.model = model

        if acquisition is None:
            acquisition = ExpectedImprovement(model)

        if model_updaters is None:
            model_updaters = FixedIntervalUpdater(model, 1)

        acquisition_optimizer = AcquisitionOptimizer(space)
        if batch_size == 1:
            candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)
        else:
            if not isinstance(model, IDifferentiable):
                raise ValueError('Model must implement ' + str(IDifferentiable) +
                                 ' for use with Local Penalization batch method.')
            log_acquisition = LogAcquisition(acquisition)
            candidate_point_calculator = LocalPenalizationPointCalculator(log_acquisition, acquisition_optimizer, model,
                                                                          space, batch_size)

        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updaters, loop_state)

    def get_results(self):
        return BayesianOptimizationResults(self.loop_state)


class BayesianOptimizationResults:
    def __init__(self,loop_state: LoopState):

        """
        Emukit class that takes as input the loop state and computes some results.

        :param loop_state: The loop state it its current form. Currently it only contains X and Y.
        """

        self.minimum_location = loop_state.X[np.argmin(loop_state.Y),:]
        self.minimum_value = np.min(loop_state.Y)
        self.best_found_value_per_iteration = np.minimum.accumulate(loop_state.Y)
