# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Union, Callable

import scipy
import numpy as np

from ...core import InformationSourceParameter
from ...core.acquisition import Acquisition
from ...core.interfaces import IModel
from ...core.parameter_space import ParameterSpace
from ...samplers import AffineInvariantEnsembleSampler, McmcSampler

from ..acquisitions import ExpectedImprovement
from ..interfaces import IEntropySearchModel
from .. import epmgp


class EntropySearch(Acquisition):

    def __init__(self, model: Union[IModel, IEntropySearchModel], space: ParameterSpace, sampler: McmcSampler = None,
                 num_samples: int = 100, num_representer_points: int = 50,
                 proposal_function: Callable = None, burn_in_steps: int = 50) -> None:

        """
        Entropy Search acquisition function approximates the distribution of the global
        minimum and tries to decrease its entropy. See this paper for more details:

        P. Hennig and C. J. Schuler
        Entropy search for information-efficient global optimization
        Journal of Machine Learning Research, 13, 2012

        :param model: GP model to compute the distribution of the minimum dubbed pmin.
        :param space: Domain space which we need for the sampling of the representer points
        :param sampler: mcmc sampler for representer points
        :param num_samples: integer determining how many samples to draw for each candidate input
        :param num_representer_points: integer determining how many representer points to sample
        :param proposal_function: Function that defines an unnormalized log proposal measure from which to sample the
        representer points. The default is expected improvement.
        :param burn_in_steps: integer that defines the number of burn-in steps when sampling the representer points
        """
        super().__init__()

        if not isinstance(model, IEntropySearchModel):
            raise RuntimeError("Model is not supported for Entropy Search")

        self.model = model
        self.num_representer_points = num_representer_points
        self.burn_in_steps = burn_in_steps

        if sampler is None:
            self.sampler = AffineInvariantEnsembleSampler(space)
        else:
            self.sampler = sampler

        # (unnormalized) density from which to sample the representer points to approximate pmin
        if proposal_function is None:

            ei = ExpectedImprovement(model)

            def prop_func(x):
                if len(x.shape) == 1:
                    x_ = x[None, :]
                else:
                    x_ = x

                if space.check_points_in_domain(x_):
                    return np.log(np.clip(ei.evaluate(x_)[0], 0., np.PINF))
                else:
                    return np.array([np.NINF])

            self.proposal_function = prop_func
        else:
            self.proposal_function = proposal_function

        # This is used later to calculate derivative of the stochastic part for the loss function
        # Derived following Ito's Lemma, see for example https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma
        self.W = scipy.stats.norm.ppf(np.linspace(1. / (num_samples + 1),
                                                  1 - 1. / (num_samples + 1),
                                                  num_samples))[np.newaxis, :]

        # Initialize parameters to lazily compute them once needed
        self.representer_points = None
        self.representer_points_log = None
        self.logP = None
        self.p_min_entropy = None

    def _sample_representer_points(self) -> tuple:
        """ Samples a new set of representer points from the proposal measurement"""

        repr_points, repr_points_log = self.sampler.get_samples(self.num_representer_points, self.proposal_function,
                                                                self.burn_in_steps)

        if np.any(np.isnan(repr_points_log)) or np.any(np.isposinf(repr_points_log)):
            raise RuntimeError(
                "Sampler generated representer points with invalid log values: {}".format(repr_points_log))

        # Removing representer points that have 0 probability of being the minimum
        idx_to_remove = np.where(np.isneginf(repr_points_log))[0]
        if len(idx_to_remove) > 0:
            idx = list(set(range(self.num_representer_points)) - set(idx_to_remove))
            repr_points = repr_points[idx, :]
            repr_points_log = repr_points_log[idx]

        return repr_points, repr_points_log

    def update_parameters(self) -> None:
        """
        Updates p_min parameter
        """
        self.update_pmin()

    def update_pmin(self) -> np.ndarray:
        """
        Approximates the distribution of the global optimum  p(x=x_star|D) by doing the following steps:
            - discretizing the input space by representer points sampled from a proposal measure (default EI)
            - predicting mean and the covariance matrix of these representer points
            - uses EPMGP algorithm to compute the probability of each representer point being the minimum
        """

        self.representer_points, self.representer_points_log = self._sample_representer_points()

        mu, _ = self.model.predict(self.representer_points)
        mu = np.ndarray.flatten(mu)
        var = self.model.predict_covariance(self.representer_points)

        self.logP, self.dlogPdMu, self.dlogPdSigma, self.dlogPdMudMu = epmgp.joint_min(mu, var, with_derivatives=True)
        self.logP = self.logP[:, np.newaxis]

        # Calculate the entropy of the distribution over the minimum given the current model
        self.p_min_entropy = np.sum(np.multiply(np.exp(self.logP), np.add(self.logP, self.representer_points_log)),
                                    axis=0)

        return self.logP

    def _required_parameters_initialized(self):
        """
        Checks if all required parameters are initialized.
        """
        return not (self.representer_points is None or self.representer_points_log is None or self.logP is None)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the information gain, i.e the change in entropy of p_min if we would evaluate x.

        :param x: points where the acquisition is evaluated.
        """
        if not self._required_parameters_initialized():
            self.update_pmin()

        # Check if we want to compute the acquisition function for multiple inputs
        if x.shape[0] > 1:
            results = np.zeros([x.shape[0], 1])
            for j in range(x.shape[0]):
                results[j] = self.evaluate(x[j, None, :])
            return results

        # Number of representer points locations
        N = self.logP.size

        # Evaluate innovations, i.e how much does mean and variance at the
        # representer points change if we would evaluate x
        dMdx, dVdx = self._innovations(x)

        dVdx = dVdx[np.triu(np.ones((N, N))).T.astype(bool), np.newaxis]

        dMdx_squared = dMdx.dot(dMdx.T)
        trace_term = np.sum(np.sum(
            np.multiply(self.dlogPdMudMu, np.reshape(dMdx_squared, (1, dMdx_squared.shape[0], dMdx_squared.shape[1]))),
            2), 1)[:, np.newaxis]

        # Deterministic part of change:
        deterministic_change = self.dlogPdSigma.dot(dVdx) + 0.5 * trace_term
        # Stochastic part of change:
        stochastic_change = (self.dlogPdMu.dot(dMdx)).dot(self.W)

        # Update our pmin distribution
        predicted_logP = np.add(self.logP + deterministic_change, stochastic_change)
        max_predicted_logP = np.amax(predicted_logP, axis=0)

        # normalize predictions
        max_diff = max_predicted_logP + np.log(np.sum(np.exp(predicted_logP - max_predicted_logP), axis=0))
        lselP = max_predicted_logP if np.any(np.isinf(max_diff)) else max_diff
        predicted_logP = np.subtract(predicted_logP, lselP)

        # We maximize the information gain
        H_p = np.sum(np.multiply(np.exp(predicted_logP), np.add(predicted_logP, self.representer_points_log)), axis=0)

        new_entropy = np.mean(H_p)
        entropy_change = new_entropy - self.p_min_entropy
        return np.array([[entropy_change]])

    def _innovations(self, x: np.ndarray) -> tuple:
        """
        Computes the expected change in mean and variance at the representer
        points (cf. Section 2.4 in the paper).

        :param x: candidate for which to compute the expected change in the GP
        """

        # Get the standard deviation at x without noise
        stdev_x = np.sqrt(self.model.predict_covariance(x, with_noise=False))

        # Compute the variance between the test point x and the representer points
        sigma_x_rep = self.model.get_covariance_between_points(self.representer_points, x)
        dm_rep = sigma_x_rep / stdev_x

        # Compute the deterministic innovation for the variance
        dv_rep = -dm_rep.dot(dm_rep.T)
        return dm_rep, dv_rep

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False


class MultiInformationSourceEntropySearch(EntropySearch):
    """
    Entropy search acquisition for multi-information source problems where the objective function is the output of one
    of the information sources. The other information sources provide auxiliary information about the objective function
    """
    def __init__(self, model: Union[IModel, IEntropySearchModel], space: ParameterSpace,
                 target_information_source_index: int=None, num_samples: int = 100,
                 num_representer_points: int = 50, burn_in_steps: int = 50):
        """
        :param model: Gaussian process model of the objective function that implements IEntropySearchModel
        :param space: Parameter space of the input domain
        :param target_information_source_index: The index of the information source we want to minimise
        :param num_samples: Integer determining how many samples to draw for each candidate input
        :param num_representer_points: Integer determining how many representer points to sample
        :param burn_in_steps: Integer that defines the number of burn-in steps when sampling the representer points
        """

        # Find information source parameter in parameter space
        info_source_parameter, source_idx = _find_source_parameter(space)
        self.source_idx = source_idx

        # Assume we are in a multi-fidelity setting and the highest index is the highest fidelity
        if target_information_source_index is None:
            target_information_source_index = max(info_source_parameter.domain)
        self.target_information_source_index = target_information_source_index

        # Sampler of representer points should sample x location at the target information source only so make a
        # parameter space without the information source parameter
        parameters_without_info_source = space.parameters.copy()
        parameters_without_info_source.remove(info_source_parameter)
        space_without_info_source = ParameterSpace(parameters_without_info_source)

        # Create sampler of representer points
        sampler = AffineInvariantEnsembleSampler(space_without_info_source)

        proposal_func = self._get_proposal_function(model, space)

        super().__init__(model, space, sampler, num_samples, num_representer_points, proposal_func, burn_in_steps)

    def _sample_representer_points(self):
        repr_points, repr_points_log = super()._sample_representer_points()

        # Add fidelity index to representer points
        idx = np.ones((repr_points.shape[0], 1)) * self.target_information_source_index
        repr_points = np.insert(repr_points, self.source_idx, idx, axis=1)
        return repr_points, repr_points_log

    def _get_proposal_function(self, model, space):

        # Define proposal function for multi-fidelity
        ei = ExpectedImprovement(model)

        def proposal_func(x):
            x_ = x[None, :]

            # Add information source parameter into array
            idx = np.ones((x_.shape[0], 1)) * self.target_information_source_index
            x_ = np.insert(x_, self.source_idx, idx, axis=1)

            if space.check_points_in_domain(x_):
                return np.log(np.clip(ei.evaluate(x_)[0], 0., np.PINF))
            else:
                return np.array([np.NINF])

        return proposal_func


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
