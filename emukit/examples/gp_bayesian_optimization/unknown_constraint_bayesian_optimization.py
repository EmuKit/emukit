# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from enum import Enum

import numpy as np
from GPy.kern import Matern52
from GPy.models import GPRegression

from .enums import AcquisitionType
from ...bayesian_optimization.acquisitions import ExpectedImprovement, NegativeLowerConfidenceBound, \
    ProbabilityOfImprovement
from ...bayesian_optimization.loops import UnknownConstraintBayesianOptimizationLoop
from ...core.parameter_space import ParameterSpace
from ...model_wrappers.gpy_model_wrappers import GPyModelWrapper


class OptimizerType(Enum):
    LBFGS = 1


class UnknownConstraintGPBayesianOptimization(UnknownConstraintBayesianOptimizationLoop):
    def __init__(self, variables_list: list, X: np.array, Y: np.array, Yc: np.array, noiseless: bool = False,
                 acquisition_type: AcquisitionType = AcquisitionType.EI, normalize_Y: bool = True,
                 acquisition_optimizer_type: OptimizerType = OptimizerType.LBFGS,
                 batch_size: int = 1,
                 model_update_interval: int = int(1)) -> None:

        """
        Class to run Bayesian optimization with unknown contraints with GPyRegression model.

        Dependencies:
            GPy (https://github.com/SheffieldML/GPy)

        :param variables_list: list containing the definition of the variables of the input space.
        :param noiseless:  determines whether the objective function is noisy or not
        :param X: initial input values where the objective has been evaluated.
        :param Y: initial output values where the objective has been evaluated.
        :param Yc: initial output values where the constraint has been evaluated.
        :param acquisition_type: type of acquisition to use during optimization.
            - EI: Expected improvement
            - PI: Probability of improvement
            - NLCB: Negative lower confidence bound
        :param normalize_Y: whether the outputs of Y are normalized in the model.
        :param acquisition_optimizer_type: selects the type of optimizer of the acquisition.
            - LBFGS: uses L-BFGS with multiple initializations.
        :param model_update_interval: interval of interactions in which the model is updated.
        :batch_size: interval of interactions in which the model is updated.
        """

        self.variables_list = variables_list
        self.noiseless = noiseless
        self.X = X
        self.Y = Y
        self.Yc = Yc
        self.acquisition_type = acquisition_type
        self.normalize_Y = normalize_Y
        self.acquisition_optimizer_type = acquisition_optimizer_type
        self.model_update_interval = model_update_interval
        self.batch_size = batch_size

        # 1. Crete the internal object to handle the input space
        self.space = ParameterSpace(variables_list)

        # 2. Select the models to use in the optimization
        self._model_chooser()
        self._model_chooser_constraint()

        # 3. Select the acquisition function
        self._acquisition_chooser()

        super().__init__(model_objective=self.model_objective, model_constraint=self.model_constraint, space=self.space,
                         acquisition=self.acquisition, batch_size=self.batch_size)

    def _model_chooser(self):
        """ Initialize the model used for the objective function """
        kernel = Matern52(len(self.variables_list), variance=1., ARD=False)
        gpmodel = GPRegression(self.X, self.Y, kernel)
        gpmodel.optimize()
        self.model_objective = GPyModelWrapper(gpmodel)
        if self.noiseless:
            gpmodel.Gaussian_noise.constrain_fixed(0.001)
        self.model_objective = GPyModelWrapper(gpmodel)

    def _model_chooser_constraint(self):
        """ Initialize the model used for the constraint """
        kernel = Matern52(len(self.variables_list), variance=1., ARD=False)
        gpmodel = GPRegression(self.X, self.Yc, kernel)
        gpmodel.optimize()
        self.model_constraint = GPyModelWrapper(gpmodel)
        if self.noiseless:
            gpmodel.Gaussian_noise.constrain_fixed(0.001)
        self.model_constraint = GPyModelWrapper(gpmodel)

    def _acquisition_chooser(self):
        """ Select the acquisition function used in the optimization """
        if self.acquisition_type is AcquisitionType.EI:
            self.acquisition = ExpectedImprovement(self.model_objective)
        elif self.acquisition_type is AcquisitionType.PI:
            self.acquisition = ProbabilityOfImprovement(self.model_objective)
        elif self.acquisition_type is AcquisitionType.NLCB:
            self.acquisition = NegativeLowerConfidenceBound(self.model_objective)

    def suggest_new_locations(self):
        """ Returns one or a batch of locations without evaluating the objective """
        return self.candidate_point_calculator.compute_next_points(self.loop_state)[0].X
