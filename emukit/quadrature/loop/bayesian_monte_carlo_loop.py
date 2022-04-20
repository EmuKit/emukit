# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import emukit.core.loop.model_updaters
from ...core.loop import FixedIntervalUpdater, ModelUpdater, OuterLoop
from ...core.loop.loop_state import create_loop_state
from ...core.parameter_space import ParameterSpace
from emukit.quadrature.loop.point_calculators.quadrature_point_calculators import BayesianMonteCarloPointCalculator
from ..methods import WarpedBayesianQuadratureModel


class BayesianMonteCarlo(OuterLoop):
    """The loop for Bayesian Monte Carlo (BMC).


    Nodes are samples from the integration measure.
    Implemented as described in Section 2.1 of the paper.

    .. rubric:: References

    C.E. Rasmussen and Z. Ghahramani, Bayesian Monte Carlo,
    Advances in Neural Information Processing Systems 15 (NeurIPS) 2003


    .. note::
        The BMC point calculator does not depend on past observations. Thus, running this BQ loop
        should be equivalent to sampling all points with MC from the measure,
        evaluating them as batch and then fitting a model to them.
        The purpose of this loop is convenience, as it can be used with the same interface
        as the active and adaptive learning schemes where point acquisition depends explicitly or implicitly
        (through hyperparameters) on the previous evaluations.

        Hint: The default :attr:`model_updater` :class:`FixedIntervalUpdater` updates and optimizes the model
        after each new sample. Since the sampling scheme of BMC does not depend on the model, alternatively,
        the dummy updater :class:`NoopModelUpdater` may be used which does not update the model.
        This may save compute time. However, the model then needs to be updated manually after the loop ran:
        i) the collected nodes are stored in ``model.loop_state``.
        ii) call ``model.set_data(loop_state.X, loop_state.Y)``
        iii) call ``model.optimize()``.

    .. seealso::
        * :class:`emukit.quadrature.loop.point_calculators.BayesianMonteCarloPointCalculator`
        * :class:`emukit.core.loop.model_updaters.NoopModelUpdater`

    :param model: A warped Bayesian quadrature model.
    :param model_updater: Defines how and when the quadrature model is updated if new data arrives,
                          defaults to :class:`FixedIntervalUpdater`.


    """

    def __init__(self, model: WarpedBayesianQuadratureModel, model_updater: ModelUpdater = None):
        if model_updater is None:
            model_updater = FixedIntervalUpdater(model, 1)

        space = ParameterSpace(model.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
        candidate_point_calculator = BayesianMonteCarloPointCalculator(model, space)
        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updater, loop_state)

        self.model = model
