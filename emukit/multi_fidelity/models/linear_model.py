# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Contains linear models
"""

import GPy
import numpy as np


class GPyLinearMultiFidelityModel(GPy.core.GP):
    """
    A thin wrapper around GPy.core.GP that does some input checking and provides a default likelihood
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, kernel: GPy.kern.Kern, n_fidelities: int,
                 likelihood: GPy.likelihoods.Likelihood=None):
        """

        :param X: Training data features with fidelity input appended as last column
        :param Y: Training data targets
        :param kernel: Multi-fidelity kernel
        :param n_fidelities: Number of fidelities in problem
        :param likelihood: GPy likelihood object.
                           Defaults to MixedNoise which has different noise levels for each fidelity
        """

        # Input checks
        if not isinstance(X, np.ndarray):
            raise ValueError('X should be an array')

        if not isinstance(Y, np.ndarray):
            raise ValueError('Y should be an array')

        if X.ndim != 2:
            raise ValueError('X should be 2d')

        if Y.ndim != 2:
            raise ValueError('Y should be 2d')

        if np.any(X[:, -1] >= n_fidelities):
            raise ValueError('One or more points has a higher fidelity index than number of fidelities')

        # Make default likelihood as different noise for each fidelity
        if likelihood is None:
            likelihood = GPy.likelihoods.mixed_noise.MixedNoise(
                [GPy.likelihoods.Gaussian(variance=1.) for _ in range(n_fidelities)])
        y_metadata = {'output_index': X[:, -1].astype(int)}
        super().__init__(X, Y, kernel, likelihood, Y_metadata=y_metadata)
