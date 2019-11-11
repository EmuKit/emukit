# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.linalg import lapack
from typing import Tuple

from ...core.acquisition import Acquisition
from ...quadrature.methods import VanillaBayesianQuadrature


class BayesianMonteCarlo(Acquisition):
    """
    This acquisition function samples from the probability distribution defined by the integration measure.
    If the integration measure is the standard Lebesque measure, then this acquisition function samples uniformly in a
    box defined by the model (reasonable box).

    Note that the acquisition function does not depend at all on past observations. Thus it is equivalent to sampling
    all points and then fit the model to them. The purpose of the acquisition function is this convenience, as it can be
    used with the same interface as the active and adaptive learning schemes that depend explicitly or implicitly
    (through hyperparameters) on the previous evaluations.
    """

    def __init__(self, model: VanillaBayesianQuadrature):
        """
        :param model: The vanilla Bayesian quadrature model
        """
        self.model = model
        if self.model.base_gp.kern.measure is not None:
            if not self.model.base_gp.kern.measure.can_sample:
                raise ValueError("The given probability measure has no method 'get_samples', but Bayesian Monte Carlo "
                                 "requires sampling capability.")

    def has_gradients(self) -> bool:
        return False

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the acquisition function at x.

        :param x: (n_points x input_dim) locations where to evaluate
        :return: (n_points x 1) the acquisition function value at x
        """
        num_points, D = x.shape
        if self.model.base_gp.kern.measure is None:
            bounds = self.model.reasonable_box_bounds.bounds
            samples = np.reshape(np.random.rand(num_points * D), [num_points, D])
            for d in range(D):
                samples[:, d] = samples[:, d] * (bounds[d][1] - bounds[d][0]) + bounds[d][0]
            return samples
        else:
            return self.model.base_gp.kern.measure.get_samples(num_points)
