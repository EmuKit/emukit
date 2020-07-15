# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Union, Callable

import scipy
import numpy as np
from scipy.stats import norm
from scipy.optimize import bisect
from scipy.integrate import simps

from ...core import InformationSourceParameter
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
        N = self.model.X.shape[0]

        random_design = RandomDesign(self.space)
        grid = random_design.get_samples(self.grid_size)
        fmean, fvar = self.model.predict(np.vstack([self.model.X, grid]), include_likelihood=False)
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


class MUMBO(MaxValueEntropySearch):
    def __init__(self, model: Union[IModel, IEntropySearchModel], space: ParameterSpace,
                 target_information_source_index: int = None, num_samples: int = 10,
                 grid_size: int = 5000) -> None:

        """
        MUMBO acquisition function approximates the distribution of the value at the global
        minimum and tries to decrease its entropy.
        This is a multi-fidelity/multi-task extension of max-value entropy search (MES) suitiable
        for multi-information source problems where the objective function is the output of one
        of the information sources. The other information sources provide auxiliary
        information about the objective function
        See this paper for more details:
        Moss et al.
        MUMBO: MUlti-task Max-value Bayesian Optimsiation
        ECML 2020

        :param model: GP model to compute the distribution of the minimum dubbed pmin.
        :param space: Domain space which we need for the sampling of the representer points
        :param target_information_source_index: The index of the information source we want to minimise
        :param num_samples: integer determining how many samples to draw of the minimum (does not need to be large)
        :param grid_size: number of random locations in grid used to fit the gumbel distribution and approximately generate
        the samples of the minimum (recommend scaling with problem dimension, i.e. 10000*d)
        """

        if not isinstance(model, IEntropySearchModel):
            raise RuntimeError("Model is not supported for MES")

        # Find information source parameter in parameter space
        info_source_parameter, source_idx = _find_source_parameter(space)
        self.source_idx = source_idx

        # Assume we are in a multi-fidelity setting and the highest index is the highest fidelity
        if target_information_source_index is None:
            target_information_source_index = max(info_source_parameter.domain)
        self.target_information_source_index = target_information_source_index

        self.model = model
        self.space = space
        self.num_samples = num_samples
        self.grid_size = grid_size

        # Initialize parameters to lazily compute them once needed
        self.mins = None

        super().__init__(model, space, num_samples, grid_size)

    def update_parameters(self):
        # apply gumbel sampling to obtain samples of y* from target information source
        # we approximate Pr(y*^hat<y) by Gumbel(alpha,beta)
        # generate grid
        N = self.model.X.shape[0]

        random_design = RandomDesign(self.space)
        grid = random_design.get_samples(self.grid_size)
        sample_points = np.vstack([self.model.X, grid])
        # remove current fidelity index from sample
        sample_points = np.delete(sample_points, self.source_idx, axis=1)
        # Add target fidelity index to sample
        idx = np.ones((sample_points.shape[0])) * self.target_information_source_index
        sample_points = np.insert(sample_points, self.source_idx, idx, axis=1)
        fmean, fvar = self.model.predict(sample_points)
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

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the information gain, i.e the change in entropy of p_min if we would evaluate x.
        :param x: points where the acquisition is evaluated.
        """
        if not self._required_parameters_initialized():
            self.update_parameters()

        # need model predictions at x and at the  x evaluated on the target fidelity (x_target_fidelity)
        # remove current fidelity index from sample
        x_target_fidelity = np.delete(x, self.source_idx, axis=1)
        # Add target fidelity index to sample
        idx = np.ones((x.shape[0])) * self.target_information_source_index
        x_target_fidelity = np.insert(x_target_fidelity, self.source_idx, idx, axis=1)

        # get predicted means and variances and pair-wise covariance
        fmean, fvar = self.model.predict(x)
        fsd = np.sqrt(fvar)
        targetmean, targetvar = self.model.gpy_model.predict(x_target_fidelity, include_likelihood=False)
        targetsd = np.sqrt(targetvar)
        # faster to do for loop rather than vectorize to avoid unecessary between term covariance calculations
        covs = [self.model.get_covariance_between_points(x[i].reshape(1, -1), x_target_fidelity[i].reshape(1, -1)) for i in range(0, x.shape[0])]
        covs = np.array(covs).reshape(-1, 1)

        # convert to pair-wise correlations
        corrs = covs / (fsd*targetsd)
        # correct for some potential rounding errors
        corrs = np.clip(corrs, -1, 1)
        corrs[x[:, -1] == self.source_idx] = 1

        # calculate moments of extended skew Gaussian distributions
        gammas = (self.mins-targetmean)/targetsd
        denom = 1 / (1 - norm.cdf(gammas))
        # need to account for numerical instability
        denom[denom == np.inf] = 0
        means = corrs * (norm.pdf(gammas)) * denom
        Vars = 1 + corrs * (means) * (gammas - norm.pdf(gammas) * denom)
        Vars[Vars <= 0] = 0

        # get upper limits for numerical integration
        upper_lim = means + 8 * np.sqrt(Vars)
        lower_lim = means - 8 * np.sqrt(Vars)

        # perform numerical integrations
        approx = np.array([approx_int(corrs[i],
                                 gammas[i],
                                 upper_lim[i],
                                 lower_lim[i]) for i in range(0, corrs.shape[0])]).reshape(-1, 1)
        return (0.5*np.log(2*np.pi*np.e)-approx).reshape(-1, 1)


def approx_int(corr, gamma, upper_lim, lower_lim):
    # helper function to numerically approx differential entropy of ESG in MUMBO
    # if corr is 1 then can just calc exactly (like MES)
    if corr == 1:
        term1 = 0.5 * np.log(2 * np.pi * np.e)
        term2 = np.log(1 - norm.cdf(gamma))
        term3 = gamma * norm.pdf(gamma) / (2 * (1 - norm.cdf(gamma)))
        return np.mean(term1 + term2 + term3)
    else:
        denom1 = 1 / ((np.sqrt(1 - corr ** 2)))
        denom2 = 1 / (1 - norm.cdf(gamma))
        # need to account for numerical instability
        denom2[denom2 == np.inf] = 0
        where_are_NaNs = np.isnan(denom2)
        denom2[where_are_NaNs] = 0
        z = np.linspace(np.min(lower_lim), np.max(upper_lim), num=5000).reshape(-1, 1)
        pdf = norm.pdf(z) * (1 - norm.cdf((gamma - corr * z) * denom1)) * denom2
        fun = - pdf * np.log(pdf, out=np.zeros_like(pdf), where=(pdf != 0))
        # perform integration over ranges
        integral = simps(fun.T, z.T).reshape(-1, 1)
        return np.mean(integral)


def _find_source_parameter(space):
    # Find information source parameter in parameter space
    info_source_parameter = None
    source_idx = None
    for i, param in enumerate(space.parameters):
        if isinstance(param, InformationSourceParameter):
            info_source_parameter = param
            source_idx = i

    if info_source_parameter is None:
        raise ValueError('No information source parameter found in the parameter space')

    return info_source_parameter, source_idx
