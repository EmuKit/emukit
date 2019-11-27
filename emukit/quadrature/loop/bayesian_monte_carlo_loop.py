# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from ...core.loop.loop_state import create_loop_state
from ...core.loop import OuterLoop, FixedIntervalUpdater, ModelUpdater
from ...core.parameter_space import ParameterSpace
from ..methods import WarpedBayesianQuadratureModel
from ..loop.quadrature_point_calculators import BayesianMonteCarloPointCalculator


class BayesianMonteCarlo(OuterLoop):
    """
    This loop implements Bayesian Monte Carlo with simple random sampling.

    Nodes are samples from the probability distribution defined by the integration measure.
    If the integration measure is the standard Lebesgue measure, then the nodes are sampled uniformly in a
    box defined by the model (reasonable box).

    C.E. Rasmussen and Z. Ghahramani
    `Bayesian Monte Carlo' Advances in Neural Information Processing Systems 15 (NeurIPS) 2003

    Note: implemented as described in Section 2.1 of the paper.

    Note that the sampling scheme does not depend at all on past observations. Thus it is equivalent to sampling
    all points and then fit the model to them. The purpose of the loop is convenience, as it can be used with the same
    interface as the active and adaptive learning schemes that depend explicitly or implicitly (through hyperparameters)
    on the previous evaluations.
    """
    def __init__(self, model: WarpedBayesianQuadratureModel, model_updater: ModelUpdater=None):
        """
        :param model: a warped Bayesian quadrature method, e.g., VanillaBayesianQuadrature
        :param model_updater: Defines how and when the quadrature model is updated if new data arrives. Defaults to
        FixedIntervalUpdater.

        Hint: The default model_updater `FixedIntervalUpdater' updates and optimizes the model after each new sample.
        Since the sampling scheme of Bayesian Monte Carlo does not depend on the model, alternatively, the dummy updater
        NoopModelUpdater may be used which does not update the model. This may save compute time. However, the model
        then needs to be updated manually after the loop ran: i) the collected nodes are stored in model.loop_state.
        ii) call model.set_data(loop_state.X, loop_state.Y) iii) call model.optimize().
        """
        if model_updater is None:
            model_updater = FixedIntervalUpdater(model, 1)

        space = ParameterSpace(model.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
        candidate_point_calculator = BayesianMonteCarloPointCalculator(model, space)
        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updater, loop_state)

        self.model = model
