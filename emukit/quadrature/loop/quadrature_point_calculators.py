# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ...core.parameter_space import ParameterSpace
from ...core.loop.loop_state import LoopState
from ...core.loop.candidate_point_calculators import CandidatePointCalculator
from ..methods.warped_bq_model import WarpedBayesianQuadratureModel
from ...core.optimization.context_manager import ContextManager


class BayesianMonteCarloPointCalculator(CandidatePointCalculator):
    """
    This point calculator implements Bayesian Monte Carlo with simple random sampling.

    The point calculator samples from the probability distribution defined by the integration measure.
    If the integration measure is the standard Lebesgue measure, then this point calculator samples uniformly in a
    box defined by the model (reasonable box).

    C.E. Rasmussen and Z. Ghahramani
    `Bayesian Monte Carlo' Advances in Neural Information Processing Systems 15 (NeurIPS) 2003

    Implemented as described in Section 2.1 of the paper.

    Note that the point calculator does not depend at all on past observations. Thus it is equivalent to sampling
    all points and then fit the model to them. The purpose of the point calculator is convenience, as it can be
    used with the same interface as the active and adaptive learning schemes that depend explicitly or implicitly
    (through hyperparameters) on the previous evaluations.
    """
    def __init__(self, model: WarpedBayesianQuadratureModel, parameter_space: ParameterSpace):
        """
        :param model: A warped Bayesian quadrature model
        """
        self.parameter_space = parameter_space
        self.model = model

        # if measure is probability measure, check if it has sampling capabilities
        if self.model.base_gp.kern.measure is not None:
            if not self.model.base_gp.kern.measure.can_sample:
                raise ValueError("The given probability measure does not support sampling, but Bayesian Monte Carlo "
                                 "requires sampling capability.")

    def compute_next_points(self, loop_state: LoopState, context: dict=None) -> np.ndarray:
        """
        :param loop_state: Object that contains current state of the loop
        :param context: Contains variables to fix and the values to fix them to. The dictionary key is the parameter
        name and the value is the value to fix the parameter to.
        :return: (1 x n_dims) array of next inputs to evaluate the function at
        """

        # Lebesgue measure
        if self.model.base_gp.kern.measure is None:
            if context is None:
                return self.parameter_space.sample_uniform(1)
            else:
                context_manager = ContextManager(self.parameter_space, context)
                samples = context_manager.contextfree_space.sample_uniform(1)
                return context_manager.expand_vector(samples)

        # probability measure
        else:
            if context is None:
                return self.model.base_gp.kern.measure.get_samples(1)
            else:
                context_manager = ContextManager(self.parameter_space, context)
                samples = self.model.base_gp.kern.measure.get_samples(1, context_manager)
                return samples
