# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.bayesian_optimization.acquisitions.log_acquisition import LogAcquisition
from emukit.bayesian_optimization.local_penalization_calculator import LocalPenalizationPointCalculator
from emukit.core.interfaces import IDifferentiable
from emukit.core.loop.loop_state import create_loop_state
from ...core.loop import OuterLoop, Sequential, FixedIntervalUpdater, LoopState, \
    UserFunctionResult, ModelUpdater, CandidatePointCalculator
from ...core.optimization import AcquisitionOptimizer
from ...core.parameter_space import ParameterSpace
from ...core.acquisition import Acquisition
from ...core.interfaces import IModel

from ..acquisitions.expected_improvement import ExpectedImprovement


class BayesianOptimizationLoop(OuterLoop):
    def __init__(self, model: IModel, space: ParameterSpace, acquisition: Acquisition = None, batch_size: int = 1,
                 model_updater: ModelUpdater = None):

        """
        Emukit class that implement a loop for building modular Bayesian optimization

        :param model: The model that approximates the underlying function
        :param space: Input space where the optimization is carried out.
        :param acquisition: The acquisition function that will be used to collect new points (default, EI). If batch
                            size is greater than one, this acquisition must output positive values only. 
        :param batch_size: How many points to evaluate in one iteration of the optimization loop. Defaults to 1.
        :param model_updater: Defines how and when the model is updated if new data arrives.
                              Defaults to updating hyper-parameters every iteration.
        """

        if acquisition is None:
            acquisition = ExpectedImprovement(model)

        if model_updater is None:
            model_updater = FixedIntervalUpdater(model, 1)

        acquisition_optimizer = AcquisitionOptimizer(space)
        if batch_size == 1:
            candidate_point_calculator = Sequential(acquisition, acquisition_optimizer)
        else:
            if not isinstance(model, IDifferentiable):
                raise ValueError('Model must implement ' + str(IDifferentiable) +
                                 ' for use with Local Penalization batch method.')
            log_acquisition = LogAcquisition(acquisition)
            candidate_point_calculator = LocalPenalizationPointCalculator(log_acquisition, acquisition_optimizer, model,
                                                                          space, batch_size)

        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updater, loop_state)
