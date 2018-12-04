# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from emukit.core.loop.loop_state import create_loop_state
from emukit.core.loop import OuterLoop, SequentialPointCalculator, FixedIntervalUpdater, ModelUpdater
from emukit.core.optimization import AcquisitionOptimizer
from emukit.core.parameter_space import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.quadrature.acquisitions import IntegralVarianceReduction


class VanillaBayesianQuadratureLoop(OuterLoop):
    def __init__(self, model: VanillaBayesianQuadrature, acquisition: Acquisition = None,
                 model_updater: ModelUpdater = None):
        """
        The loop for vanilla Bayesian Quadrature

        :param model: the vanilla Bayesian quadrature method
        :param acquisition: The acquisition function that is be used to collect new points.
        default, IntegralVarianceReduction
        :param model_updater: Defines how and when the quadrature model is updated if new data arrives.
                              Defaults to updating hyper-parameters every iteration.
        """

        if acquisition is None:
            acquisition = IntegralVarianceReduction(model)

        if model_updater is None:
            model_updater = FixedIntervalUpdater(model, 1)

        space = ParameterSpace(model.integral_bounds.convert_to_list_of_continuous_parameters())
        acquisition_optimizer = AcquisitionOptimizer(space)
        candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)
        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updater, loop_state)

        self.model = model
