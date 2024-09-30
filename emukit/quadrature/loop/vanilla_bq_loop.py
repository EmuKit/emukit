# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from ...core.acquisition import Acquisition
from ...core.loop import FixedIntervalUpdater, ModelUpdater, SequentialPointCalculator
from ...core.optimization import AcquisitionOptimizerBase, GradientAcquisitionOptimizer
from ...core.parameter_space import ParameterSpace
from ..acquisitions import IntegralVarianceReduction
from ..methods import VanillaBayesianQuadrature
from .bq_loop_state import create_bq_loop_state
from .bq_outer_loop import QuadratureOuterLoop


class VanillaBayesianQuadratureLoop(QuadratureOuterLoop):
    """The loop for standard ('vanilla') Bayesian Quadrature.

    .. seealso::
        :class:`emukit.quadrature.methods.VanillaBayesianQuadrature`

    :param model: The vanilla Bayesian quadrature model.
    :param acquisition: The acquisition function that is used to collect new points,
                        defaults to integral-variance-reduction.
    :param model_updater: Defines how and when the BQ model is updated if new data arrives.
                          Defaults to updating hyper-parameters after every iteration.
    :param acquisition_optimizer: Optimizer selecting next evaluation points by maximizing acquisition.
                                  Gradient based optimizer is used if None. Defaults to None.
    """

    def __init__(
        self,
        model: VanillaBayesianQuadrature,
        acquisition: Acquisition = None,
        model_updater: ModelUpdater = None,
        acquisition_optimizer: AcquisitionOptimizerBase = None,
    ):
        if acquisition is None:
            acquisition = IntegralVarianceReduction(model)

        if model_updater is None:
            model_updater = FixedIntervalUpdater(model, 1)

        space = ParameterSpace(model.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
        if acquisition_optimizer is None:
            acquisition_optimizer = GradientAcquisitionOptimizer(space)
        candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)
        loop_state = create_bq_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updater, loop_state)

        self.model = model
