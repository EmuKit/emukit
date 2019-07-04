# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Union
from ...bayesian_optimization.acquisitions import ExpectedImprovement, ProbabilityOfFeasibility
from ...core.acquisition import Acquisition
from ...core.interfaces import IModel, IDifferentiable
from ...core.loop import FixedIntervalUpdater, OuterLoop, SequentialPointCalculator
from ...core.loop.loop_state import create_loop_state
from ...core.optimization import AcquisitionOptimizerBase
from ...core.optimization import GradientAcquisitionOptimizer
from ...core.parameter_space import ParameterSpace
from ..acquisitions.log_acquisition import LogAcquisition
from ..local_penalization_calculator import LocalPenalizationPointCalculator


class UnknownConstraintBayesianOptimizationLoop(OuterLoop):
    def __init__(self, space: ParameterSpace, model_objective: Union[IModel, IDifferentiable],
                 model_constraint: Union[IModel, IDifferentiable], acquisition: Acquisition = None,
                 update_interval: int = 1, batch_size: int = 1):

        """
        Emukit class that implements a loop for building Bayesian optimization with an unknown constraint.
        For more information see:

        Michael A. Gelbart, Jasper Snoek, and Ryan P. Adams,
        Bayesian Optimization with Unknown Constraints,
        https://arxiv.org/pdf/1403.5607.pdf

        :param space: Input space where the optimization is carried out.
        :param model_objective: The model that approximates the underlying objective function
        :param model_constraint: The model that approximates the unknown constraints
        :param acquisition: The acquisition function for the objective function (default, EI).
        :param update_interval:  Number of iterations between optimization of model hyper-parameters. Defaults to 1.
        :param batch_size: How many points to evaluate in one iteration of the optimization loop. Defaults to 1.
        """

        if not np.all(np.isclose(model_objective.X, model_constraint.X)):
            raise ValueError('Emukit currently only supports identical '
                             'training inputs for the constrained and objective model')

        if acquisition is None:
            acquisition = ExpectedImprovement(model_objective)
 
        acquisition_constraint = ProbabilityOfFeasibility(model_constraint)

        acquisition_constrained = acquisition * acquisition_constraint

        model_updater_objective = FixedIntervalUpdater(model_objective, update_interval)
        model_updater_constraint = FixedIntervalUpdater(model_constraint, update_interval,
                                                        lambda state: state.Y_constraint)

        acquisition_optimizer = GradientAcquisitionOptimizer(space)
        if batch_size == 1:
            candidate_point_calculator = SequentialPointCalculator(acquisition_constrained, acquisition_optimizer)
        else:
            log_acquisition = LogAcquisition(acquisition_constrained)
            candidate_point_calculator = LocalPenalizationPointCalculator(log_acquisition, acquisition_optimizer,
                                                                          model_objective, space, batch_size)

        loop_state = create_loop_state(model_objective.X, model_objective.Y, Y_constraint=model_constraint.Y)

        super(UnknownConstraintBayesianOptimizationLoop, self).__init__(candidate_point_calculator,
                                                                    [model_updater_objective, model_updater_constraint],
                                                                    loop_state)
