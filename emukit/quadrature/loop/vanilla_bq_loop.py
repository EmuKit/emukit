# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from ...core.loop.loop_state import create_loop_state
from ...core.loop import OuterLoop, SequentialPointCalculator, FixedIntervalUpdater, ModelUpdater
from ...core.optimization import AcquisitionOptimizerBase, GradientAcquisitionOptimizer
from ...core.parameter_space import ParameterSpace
from ...core.acquisition import Acquisition
from ..methods import VanillaBayesianQuadrature
from ..acquisitions import IntegralVarianceReduction


class VanillaBayesianQuadratureLoop(OuterLoop):
    def __init__(self, model: VanillaBayesianQuadrature, acquisition: Acquisition = None,
                 model_updater: ModelUpdater = None, acquisition_optimizer: AcquisitionOptimizerBase = None):
        """
        The loop for vanilla Bayesian Quadrature

        :param model: the vanilla Bayesian quadrature method
        :param acquisition: The acquisition function that is used to collect new points.
        default, IntegralVarianceReduction
        :param model_updater: Defines how and when the quadrature model is updated if new data arrives.
                              Defaults to updating hyper-parameters every iteration.
        :param acquisition_optimizer: Optimizer selecting next evaluation points by maximizing acquisition.
                                      Gradient based optimizer is used if None. Defaults to None.
        """

        if acquisition is None:
            acquisition = IntegralVarianceReduction(model)

        if model_updater is None:
            model_updater = FixedIntervalUpdater(model, 1)

        space = ParameterSpace(model.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
        if acquisition_optimizer is None:
            acquisition_optimizer = GradientAcquisitionOptimizer(space)
        candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)
        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updater, loop_state)

        self.model = model
