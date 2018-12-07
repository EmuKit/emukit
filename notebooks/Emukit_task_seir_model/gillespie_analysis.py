# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple
from scipy.stats import gamma

from .sir_gillespie import SIRGillespie


class AlphaPrior:
    """ Defines possible priors over the parameter alpha """
    def __init__(self, name):
        self.name = name

    def evaluate(self, alpha: float) -> float:
        """
        :param alpha: the ratio of infection rate and recovery rate
        :return: the probability density at alpha
        """
        raise NotImplemented


class GammaPrior(AlphaPrior):
    """ Gamma prior on the infection/recovery rate parameter """
    def __init__(self, a: int, loc: float, scale: float):
        """
        :param a: shape parameter
        :param loc: shift the gamma distribution
        :param scale: scale parameter for controlling the width
        """
        self.a = a
        self.loc = loc
        self.scale = scale
        self.gamma = gamma(a, loc, scale)
        super(GammaPrior, self).__init__(name='gamma prior')

    def evaluate(self, alpha: float) -> float:
        """
        :param alpha: the ratio of infection rate and recovery rate
        :return: the probability density at alpha
        """
        return self.gamma.pdf(alpha)


class UniformPrior(AlphaPrior):
    """ Uniform prior on the infection/recovery rate parameter """
    def __init__(self, alpha_min: float, alpha_max: float):
        """
        :param alpha_min: left interval bound
        :param alpha_max: right interval bound
        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        super(UniformPrior, self).__init__(name='uniform prior')

    def evaluate(self, alpha: float) -> float:
        """
        :param alpha: the ratio of infection rate and recovery rate
        :return: the probability density at alpha
        """
        return 1./(self.alpha_max - self.alpha_min)


class MeanMaxInfectionGillespie:
    """
    Statistics for the time occurrence and height of the infection peak of the gillespie simulation.
    """
    def __init__(self, gillespie_model: SIRGillespie, num_gil: int, time_end: float, alpha_prior: AlphaPrior):
        """
        :param gillespie_model: a SIRGillespie model
        :param alpha_prior: the prior over alpha
        :param num_gil: number of Gillespie samples to average over
        :param time_end: end time of simulation
        """
        self.gillespie = gillespie_model
        self.alpha_prior = alpha_prior
        self.num_gil = num_gil
        self.time_end = time_end

    def evaluate_bare(self, alpha: float) -> Tuple[float, float]:
        """
        :param alpha: the ratio of infection rate and recovery rate
        :return: Estimated (mean over num_gil simulations runs) time and height of the infection peak for given alpha.
        """
        self.gillespie.model.set_alpha(alpha)
        peak_time, peak_height = self.gillespie.run_simulation_height_and_time_of_peak(self.num_gil, self.time_end)
        return peak_time, peak_height

    def evaluate(self, alpha: float) -> Tuple[float, float]:
        """
        :param alpha: the ratio of infection rate and recovery rate
        :return: Estimated (mean over num_gil simulations runs) time and height of the infection peak for given alpha
        scaled with the prior over alpha.
        """
        peak_time, peak_height = self.evaluate_bare(alpha)
        alpha_weight = self.alpha_prior.evaluate(alpha)
        return peak_time * alpha_weight, peak_height * alpha_weight

    def evaluate_time_bare(self, alpha: float) -> float:
        return self.evaluate_bare(alpha)[0]

    def evaluate_height_bare(self, alpha: float) -> float:
        return self.evaluate_bare(alpha)[1]

    def evaluate_time(self, alpha: float) -> float:
        return self.evaluate(alpha)[0]

    def evaluate_height(self, alpha: float) -> float:
        return self.evaluate(alpha)[1]


# the following functions collect specific statistics of the Gillespie simulation and are used as emukit user-functions
# in the corresponding task notebook.
def _f_height_of_peak_weighted(alpha, meanmax: MeanMaxInfectionGillespie):
    if isinstance(alpha, np.ndarray):
        return np.asarray([meanmax.evaluate_height(float(alpha_i)) for alpha_i in alpha])[:, None]
    else:
        return np.asarray([meanmax.evaluate_height(alpha)])[:, None]


def _f_time_of_peak_weighted(alpha, meanmax: MeanMaxInfectionGillespie):
    if isinstance(alpha, np.ndarray):
        return np.asarray([meanmax.evaluate_time(float(alpha_i)) for alpha_i in alpha])[:, None]
    else:
        return np.asarray([meanmax.evaluate_time(alpha)])[:, None]


def _f_height_of_peak(alpha, meanmax: MeanMaxInfectionGillespie):
    if isinstance(alpha, np.ndarray):
        return np.asarray([meanmax.evaluate_height_bare(float(alpha_i)) for alpha_i in alpha])[:, None]
    else:
        return np.asarray([meanmax.evaluate_height_bare(alpha)])[:, None]


def _f_time_of_peak(alpha, meanmax: MeanMaxInfectionGillespie):
    if isinstance(alpha, np.ndarray):
        return np.asarray([meanmax.evaluate_time_bare(float(alpha_i)) for alpha_i in alpha])[:, None]
    else:
        return np.asarray([meanmax.evaluate_time_bare(alpha)])[:, None]


def height_of_peak_weighted(meanmax: MeanMaxInfectionGillespie):
    return lambda alpha: _f_height_of_peak_weighted(alpha, meanmax)


def time_of_peak_weighted(meanmax: MeanMaxInfectionGillespie):
    return lambda alpha: _f_time_of_peak_weighted(alpha, meanmax)


def height_of_peak(meanmax: MeanMaxInfectionGillespie):
    return lambda alpha: _f_height_of_peak(alpha, meanmax)


def time_of_peak(meanmax: MeanMaxInfectionGillespie):
    return lambda alpha: _f_time_of_peak(alpha, meanmax)
