# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from ...core.loop.loop_state import create_loop_state
from ...core.loop import OuterLoop, SequentialPointCalculator, FixedIntervalUpdater, ModelUpdater
from ...core.loop.model_updaters import NoopModelUpdater
from ...core.optimization import AcquisitionOptimizerBase, GradientAcquisitionOptimizer
from ...core.parameter_space import ParameterSpace
from ...core.acquisition import Acquisition
from ...quadrature.methods import VanillaBayesianQuadrature, WarpedBayesianQuadratureModel
from ...quadrature.acquisitions import IntegralVarianceReduction
from ...quadrature.loop.quadrature_point_calculators import SimpleBayesianMonteCarloPointCalculator


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


class SimpleBayesianMonteCarlo(OuterLoop):
    """
    This loop implements Simple Bayesian Monte Carlo.

    Nodes are samples from the probability distribution defined by the integration measure.
    If the integration measure is the standard Lebesgue measure, then the nodes are sampled uniformly in a
    box defined by the model (reasonable box).

    C.E. Rasmussen and Z. Ghahramani
    `Bayesian Monte Carlo' Advances in Neural Information Processing Systems 15 (NeurIPS) 2003

    Note: implemented as described in Section 2.1 of the paper.

    Note that this method does not depend at all on past observations. Thus it is equivalent to sampling
    all points and then fit the model to them. The purpose of the loop is convenience, as it can be
    used with the same interface as the active and adaptive learning schemes that depend explicitly or implicitly
    (through hyperparameters) on the previous evaluations.
    """
    def __init__(self, model: WarpedBayesianQuadratureModel, model_updater: ModelUpdater=None):
        """
        :param model: a warped Bayesian quadrature method, e.g., VanillaBayesianQuadrature
        :param model_updater: Defines how and when the quadrature model is updated if new data arrives. Defaults to
        FixedIntervalUpdater. If the dummy updater NoopModelUpdater is used which does not update the model, the
        collected nodes are stored in the loop state. When using NoopModelUpdater, you can update the model after the
        loop ran by calling model.set_data(loop_state.X, loop_state.Y); model.optimize().
        """
        if model_updater is None:
            model_updater = FixedIntervalUpdater(model, 1)

        space = ParameterSpace(model.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
        candidate_point_calculator = SimpleBayesianMonteCarloPointCalculator(model, space)
        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updater, loop_state)

        self.model = model
