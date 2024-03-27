# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""The WSABI-L loop"""


from ...core.loop import FixedIntervalUpdater, ModelUpdater, OuterLoop, SequentialPointCalculator
from ...core.loop.loop_state import create_loop_state
from ...core.optimization import AcquisitionOptimizerBase, GradientAcquisitionOptimizer
from ...core.parameter_space import ParameterSpace
from ..acquisitions import UncertaintySampling
from ..methods import WSABIL


class WSABILLoop(OuterLoop):
    """The loop for WSABI-L.

    .. rubric:: References

    Gunter et al. 2014 *Sampling for Inference in Probabilistic Models with Fast Bayesian Quadrature*,
    Advances in Neural Information Processing Systems (NeurIPS), 27, pp. 2789–2797.

    .. seealso::
        :class:`emukit.quadrature.methods.WSABIL`

    :param model: The WSABI-L model.
    :param model_updater: Defines how and when the quadrature model is updated if new data arrives.
                          Defaults to updating hyper-parameters every iteration.
    :param acquisition_optimizer: Optimizer selecting next evaluation points by maximizing acquisition.
                                  Gradient based optimizer is used if None. Defaults to None.
    """

    def __init__(
        self, model: WSABIL, model_updater: ModelUpdater = None, acquisition_optimizer: AcquisitionOptimizerBase = None
    ):
        # WSABI-L is used with uncertainty sampling.
        acquisition = UncertaintySampling(model, measure_power=1)

        if model_updater is None:
            model_updater = FixedIntervalUpdater(model, 1)

        space = ParameterSpace(model.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
        if acquisition_optimizer is None:
            acquisition_optimizer = GradientAcquisitionOptimizer(space)
        candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)
        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updater, loop_state)

        self.model = model
