# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Union, Callable

import scipy
import numpy as np
from scipy.stats import norm
from scipy.optimize import bisect

from ...core.acquisition import Acquisition
from ...core.interfaces import IModel
from ...core.parameter_space import ParameterSpace
from ...core.initial_designs import RandomDesign

from ..interfaces import IEntropySearchModel


class MaxValueEntropySearch(Acquisition):
    def __init__(self, model: Union[IModel, IEntropySearchModel], space: ParameterSpace,
                 num_samples: int = 10, grid_size: int = 5000) -> None:

        """
        MES acquisition function approximates the distribution of the value at the global
        minimum and tries to decrease its entropy. See this paper for more details:
        Z. Wang, S. Jegelka
        Max-value Entropy Search for Efficient Bayesian Optimization
        ICML 2017

        :param model: GP model to compute the distribution of the minimum dubbed pmin.
        :param space: Domain space which we need for the sampling of the representer points
        :param num_samples: integer determining how many samples to draw of the minimum (does not need to be large)
        :param grid_size: number of random locations in grid used to fit the gumbel distribution and approximately generate
        the samples of the minimum (recommend scaling with problem dimension, i.e. 10000*d)
        """
        super().__init__()

        if not isinstance(model, IEntropySearchModel):
            raise RuntimeError("Model is not supported for MES")

        self.model = model
        self.space = space
        self.num_samples = num_samples
        self.grid_size = grid_size

        # Initialize parameters to lazily compute them once needed
        self.mins = None

    def update_parameters(self):
        # apply gumbel sampling to obtain samples from y*
        # we approximate Pr(y*^hat<y) by Gumbel(alpha,beta)
        # generate grid
        N = self.model.model.X.shape[0]

        random_design = RandomDesign(self.space)
        grid = random_design.get_samples(self.grid_size)
        fmean, fvar = self.model.model.predict(np.vstack([self.model.model.X, grid]), include_likelihood=False)
        fsd = np.sqrt(fvar)
        idx = np.argmin(fmean[:N])

        # scaling so that gumbel scale is proportional to IQ range of cdf Pr(y*<z)
        # find quantiles Pr(y*<y1)=r1 and Pr(y*<y2)=r2
        right = fmean[idx].flatten()
        left = right
        probf = lambda x: np.exp(np.sum(norm.logcdf(-(x - fmean) / fsd), axis=0))
        i = 0
        while probf(left) < 0.75:
            left = 2. ** i * np.min(fmean - 5. * fsd) + (1. - 2. ** i) * right
            i += 1
        i = 0
        while probf(right) > 0.25:
            right = -2. ** i * np.min(fmean - 5. * fsd) + (1. + 2. ** i) * fmean[idx].flatten()
            i += 1

        # Binary search for 3 percentiles
        q1, med, q2 = map(lambda val: bisect(lambda x: probf(x) - val, left, right, maxiter=10000, xtol=0.00001),
                            [0.25, 0.5, 0.75])

        # solve for gumbel params
        beta = (q1 - q2) / (np.log(np.log(4. / 3.)) - np.log(np.log(4.)))
        alpha = med + beta * np.log(np.log(2.))

        # sample K length vector from unif([0,1])
        # return K Y* samples
        self.mins = -np.log(-np.log(np.random.rand(self.num_samples))) * beta + alpha

    def _required_parameters_initialized(self):
        """
        Checks if all required parameters are initialized.
        """
        return self.mins is not None

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the information gain, i.e the change in entropy of p_min if we would evaluate x.
        :param x: points where the acquisition is evaluated.
        """
        if not self._required_parameters_initialized():
            self.update_parameters()
        fmean, fvar = self.model.predict(x)
        fsd = np.sqrt(fvar)
        gamma = (self.mins - fmean) / fsd
        f_acqu_x = np.mean(-gamma * norm.pdf(gamma) / (2 * (1 - norm.cdf(gamma))) - np.log(1 - norm.cdf(gamma)),
                            axis=1)
        return f_acqu_x.reshape(-1, 1)

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False
