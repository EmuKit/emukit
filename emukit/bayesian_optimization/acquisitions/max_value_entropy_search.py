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
        """
        MES requires acces to a sample of possible minimum values y* of the objective function.
        To build this sample we approximate the empirical c.d.f of Pr(y*<y) with a Gumbel(a,b) distribution.
        This Gumbel distribution can then be easily sampled to yield approximate samples of y*
        
        This needs to be called once at the start of each BO step.
        """

        # First we generate a random grid of locations at which to fit the Gumbel distribution
        random_design = RandomDesign(self.space)
        grid = random_design.get_samples(self.grid_size)
        # also add the locations already queried in the previous BO steps
        grid = np.vstack([self.model.X, grid])
        # Get GP posterior at these points
        fmean, fvar = self.model.predict(grid)
        fsd = np.sqrt(fvar)

        # fit Gumbel distriubtion
        a, b = _fit_gumbel(fmean, fsd)

        # sample K times from this Gumbel distribution using the inverse probability integral transform,
        # i.e. given a sample r ~ Unif[0,1] then g = a + b * log( -1 * log(1 - r)) follows g ~ Gumbel(a,b).

        uniform_samples = np.random.rand(self.num_samples)
        gumbel_samples = np.log(-1 * np.log(1 - uniform_samples)) * b + a
        self.mins = gumbel_samples

    def _required_parameters_initialized(self):
        """
        Checks if all required parameters are initialized.
        """
        return self.mins is not None

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the information gain, i.e the predicted change in entropy of p_min (the distribution
        of the minimal value of the objective function) if we evaluate x.
        :param x: points where the acquisition is evaluated.
        """
        if not self._required_parameters_initialized():
            self.update_parameters()

        # Calculate GP posterior at candidate points
        fmean, fvar = self.model.predict(x)
        fsd = np.sqrt(fvar)
        # Clip below to improve numerical stability
        fsd = np.maximum(fsd, 1e-10)

        # standardise
        gamma = (self.mins - fmean) / fsd

        minus_cdf = 1 - norm.cdf(gamma)
        # Clip  to improve numerical stability
        minus_cdf = np.clip(minus_cdf, a_min = 1e-10, a_max = 1)

        # calculate monte-carlo estimate of information gain
        f_acqu_x = np.mean(-gamma * norm.pdf(gamma) / (2 * minus_cdf) - np.log(minus_cdf), axis=1)
        return f_acqu_x.reshape(-1, 1)

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False



def _fit_gumbel(fmean, fsd):
    """ 
    Helper function to fit gumbel distribution when initialising the MES and MUMBO acquisition functions.

    The Gumbel distribution for minimas has a cumulative density function of f(y)= 1 - exp(-1 * exp((y - a) / b)), i.e. the q^th quantile is given by 
    Q(q) = a + b * log( -1 * log(1 - q)). We choose values for a and b that match the Gumbel's 
    interquartile range with that of the observed empirical cumulative density function of Pr(y*<y)
    i.e.  Pr(y* < lower_quantile)=0.25 and Pr(y* < upper_quantile)=0.75.
    """
    def probf(x: np.ndarray) -> float:
        # Build empirical CDF function
        return 1 - np.exp(np.sum(norm.logcdf(-(x - fmean) / fsd), axis=0))
    
    # initialise end-points for binary search (the choice of 5 standard deviations ensures that these are outside the IQ range)
    left = np.min(fmean - 5 * fsd)
    right = np.max(fmean + 5 * fsd)

    def binary_search(val: float) -> float:
        return bisect(lambda x: probf(x) - val, left, right, maxiter=10000)


    # Binary search for 3 percentiles
    lower_quantile, medium, upper_quantile = map(binary_search, [0.25, 0.5, 0.75])

    # solve for Gumbel scaling parameters
    b = (lower_quantile - upper_quantile) / (np.log(np.log(4. / 3.)) - np.log(np.log(4.)))
    a = medium - b * np.log(np.log(2.))

    return a, b



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

        # If not told otherwise assume we are in a multi-fidelity setting 
        # and the highest index is the highest fidelity
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
        """
        MUMBO requires acces to a sample of possible minimum values y* of the objective function.
        To build this sample we approximate the empirical c.d.f of Pr(y*<y) with a Gumbel(a,b) distribution.
        This Gumbel distribution can then be easily sampled to yield approximate samples of y*
        
        This needs to be called once at the start of each BO step.
        """        
        
        # First we generate a random grid of locations at which to fit the Gumbel distribution
        random_design = RandomDesign(self.space)
        grid = random_design.get_samples(self.grid_size)
        # also add the locations already queried in the previous BO steps
        grid = np.vstack([self.model.X, grid])
        # remove current fidelity index from sample
        grid = np.delete(grid, self.source_idx, axis=1)
        # Add objective function fidelity index to sample
        idx = np.ones((grid.shape[0])) * self.target_information_source_index
        grid = np.insert(grid, self.source_idx, idx, axis=1)
        # Get GP posterior at these points
        fmean, fvar = self.model.predict(grid)
        fsd = np.sqrt(fvar)

        # fit Gumbel distriubtion
        a, b = _fit_gumbel(fmean, fsd)

        # sample K times from this Gumbel distribution
        uniform_samples = np.random.rand(self.num_samples)
        gumbel_samples = -np.log(-np.log(uniform_samples)) * b + a
        self.mins = gumbel_samples


    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the information gain, i.e the change in entropy of p_min (the distribution
        of the minimal value of the objective function) if we would evaluate x.
        :param x: points where the acquisition is evaluated.
        """
        if not self._required_parameters_initialized():
            self.update_parameters()


        # Calculate GP posterior at candidate points
        fmean, fvar = self.model.predict(x)
        fsd = np.sqrt(fvar)
        # clip below for numerical stability
        fsd = np.maximum(fsd, 1e-10)

        # Also need posterior at locations with same search-space positions as x but on the objective function g
        # remove current fidelity index from sample
        x_target_fidelity = np.delete(x, self.source_idx, axis=1)
        # Add target fidelity index to sample
        idx = np.ones((x.shape[0])) * self.target_information_source_index
        x_target_fidelity = np.insert(x_target_fidelity, self.source_idx, idx, axis=1)
        gmean, gvar = self.model.predict(x_target_fidelity)
        gsd = np.sqrt(gvar)
        # clip below for numerical stability
        gsd = np.maximum(gsd, 1e-10)

        # also get pair-wise correlations between GP at x and x_target_fidelity
        # faster to do for loop rather than vectorize to avoid unecessary between term covariance calculations
        covariances = [self.model.get_covariance_between_points(x[i].reshape(1, -1), x_target_fidelity[i].reshape(1, -1)) for i in range(0, x.shape[0])]
        covariances = np.array(covariances).reshape(-1, 1)
        correlations = covariances / (fsd * gsd)
        # clip for numerical stability
        correlations = np.clip(correlations, -1, 1)

        # Calculate variance of extended skew Gaussian distributions (ESG) 
        # These will be used to define reasonable ranges for the numerical 
        # intergration of the ESG's differential entropy.
        gammas = (self.mins - fmean) / fsd
        minus_cdf = 1 - norm.cdf(gammas)
        # Clip  to improve numerical stability
        minus_cdf = np.clip(minus_cdf, a_min = 1e-10, a_max = 1)
        ESGmean = correlations * (norm.pdf(gammas)) / minus_cdf
        ESGvar = 1 + correlations * ESGmean * (gammas - norm.pdf(gammas) / minus_cdf)
        # Clip  to improve numerical stability
        ESGvar = np.maximum(ESGvar, 0)

        # get upper limits for numerical integration 
        # we need this range to contain almost all of the ESG's probability density
        # we found +-8 standard deviations provides a tight enough approximation 
        upper_limit = ESGmean + 8 * np.sqrt(ESGvar)
        lower_limit = ESGmean - 8 * np.sqrt(ESGvar)

        # perform numerical integrations 
        # build discretisation
        z = np.linspace(lower_limit, upper_limit, num=5000)
        # calculate ESG density at these points
        minus_correlations = np.sqrt(1 - correlations ** 2)
        # clip below for numerical stability
        minus_correlations = np.maximum(minus_correlations,1e-10)
        density = norm.pdf(z) * (1 - norm.cdf((gammas - correlations* z) / minus_correlations)) / minus_cdf
        # calculate point-wise entropy function contributions (carefuly where density is 0)
        entropy_function = - density * np.log(density, out=np.zeros_like(density), where=(density != 0))
        # perform integration over ranges
        approximate_entropy = simps(entropy_function.T, z.T)
        # build monte-carlo estimate over the gumbel samples
        approximate_entropy = np.mean(approximate_entropy, axis=0)

        # build MUMBO acquisition function
        f_acqu_x = (0.5*np.log(2*np.pi*np.e)-approximate_entropy)
        return f_acqu_x.reshape(-1, 1)

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
