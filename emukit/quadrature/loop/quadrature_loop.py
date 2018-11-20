# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from emukit.core.loop.loop_state import create_loop_state
from emukit.core.loop import OuterLoop, Sequential, FixedIntervalUpdater, ModelUpdater
from emukit.core.optimization import AcquisitionOptimizer
from emukit.quadrature.methods import VanillaBayesianQuadrature
# TODO: replace this with BQ acquisition
from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound
#from emukit.quadrature.acquisitions import IntegratedVarianceReduction

from emukit.core.parameter_space import ParameterSpace
from emukit.core.acquisition import Acquisition


class VanillaBayesianQuadratureLoop(OuterLoop):
    def __init__(self, model: VanillaBayesianQuadrature, space: ParameterSpace, acquisition: Acquisition = None,
                 model_updater: ModelUpdater = None):
        """
        Emukit class that implement a loop for building modular Bayesian optimization

        :param model: the vanilla Bayesian quadrature method
        :param space: the domain of the integral
        :param acquisition: The acquisition function that is be used to collect new points. default, variance reduction
        :param model_updater: Defines how and when the quadrature model is updated if new data arrives.
                              Defaults to updating hyper-parameters every iteration.
        """


        # TODO: this need to be e.g., variance reduction
        if acquisition is None:
            acquisition = NegativeLowerConfidenceBound(model)
            #acquisition = IntegratedVarianceReduction(model)

        if model_updater is None:
            model_updater = FixedIntervalUpdater(model, 1)

        acquisition_optimizer = AcquisitionOptimizer(space)
        candidate_point_calculator = Sequential(acquisition, acquisition_optimizer)
        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updater, loop_state)
