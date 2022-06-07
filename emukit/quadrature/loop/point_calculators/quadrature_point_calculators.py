# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ....core.loop.candidate_point_calculators import CandidatePointCalculator
from ....core.loop.loop_state import LoopState
from ....core.optimization.context_manager import ContextManager
from ....core.parameter_space import ParameterSpace
from ...methods.warped_bq_model import WarpedBayesianQuadratureModel


class BayesianMonteCarloPointCalculator(CandidatePointCalculator):
    """This point calculator produces Monte Carlo points from the integration measure.

    It can be used for Bayesian Monte Carlo (BMC) `[1]`_ as described in Section 2.1 of the paper.

    .. _[1]:

    .. rubric:: References

    [1] C.E. Rasmussen and Z. Ghahramani, Bayesian Monte Carlo,
    Advances in Neural Information Processing Systems 15 (NeurIPS) 2003

    .. seealso::
        :class:`emukit.quadrature.loop.BayesianMonteCarlo`

    :param model: A warped Bayesian quadrature model.
    :param parameter_space: The parameter space.

    :raises ValueError: If integration measure provided through the model does have sampling capabilities.

    """

    def __init__(self, model: WarpedBayesianQuadratureModel, parameter_space: ParameterSpace):
        self.parameter_space = parameter_space
        self.model = model

        # if measure is probability measure, check if it has sampling capabilities
        if self.model.base_gp.kern.measure is not None:
            if not self.model.base_gp.kern.measure.can_sample:
                raise ValueError(
                    "The given probability measure does not support sampling, but Bayesian Monte Carlo "
                    "requires sampling capability."
                )

    def compute_next_points(self, loop_state: LoopState, context: dict = None) -> np.ndarray:
        """Computes the next point.

        :param loop_state: Object that contains current state of the loop.
        :param context: Contains variables to fix and the values to fix them to. The dictionary key is the parameter
                        name and the value is the value to fix the parameter to.
        :return: The next point to evaluate the function at, shape (1, input_dim).
        """

        measure = self.model.base_gp.kern.measure

        # Lebesgue measure
        if measure is None:
            if context is None:
                return self.parameter_space.sample_uniform(1)
            else:
                context_manager = ContextManager(self.parameter_space, context)
                samples = context_manager.contextfree_space.sample_uniform(1)
                return context_manager.expand_vector(samples)

        # probability measure
        else:
            if context is None:
                return measure.sample(1)
            else:
                context_manager = ContextManager(self.parameter_space, context)
                samples = measure.sample(1, context_manager)
                return samples
