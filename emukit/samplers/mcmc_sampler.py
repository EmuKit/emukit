# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import emcee


class McmcSampler(object):
    def __init__(self, space):
        """
        Creates an instance of the sampler.

        Parameters:
            space - variable space
        """
        self.space = space

    def get_samples(self, n_samples, log_p_function, burn_in_steps=50):
        """
        Generates samples.

        Parameters:
            n_samples - number of samples to generate
            log_p_function - a function that returns log density for a specific sample
            burn_in_steps - number of burn-in steps for sampling

        Returns a tuple of two lists: (samples, log_p_function values for samples)
        """

        raise NotImplementedError


class AffineInvariantEnsembleSampler(McmcSampler):
    def __init__(self, space):
        """
        Creates an instance of the affine invariant ensemble sampler.

        Parameters:
            space - variable space
        """
        super(AffineInvariantEnsembleSampler, self).__init__(space)

    def get_samples(self, n_samples, log_p_function, burn_in_steps=50, n_steps=100):
        """
        Generates samples.

        Parameters:
            n_samples - number of samples to generate
            log_p_function - a function that returns log density for a specific sample
            burn_in_steps - number of burn-in steps for sampling

        Returns a tuple of two array: (samples, log_p_function values for samples)
        """
        X_init = self.space.sample_uniform(n_samples)
        sampler = emcee.EnsembleSampler(n_samples, X_init.shape[1], log_p_function)

        # Burn-In
        state = list(sampler.run_mcmc(X_init, burn_in_steps)) # compatible with both emcee 2 and 3
        samples = state[0]
        samples_log = state[1]

        # MCMC Sampling
        state = list(sampler.run_mcmc(samples, n_steps))
        samples = state[0]
        samples_log = state[1]

        # make sure we have an array of shape (n samples, space input dim)
        if len(samples.shape) == 1:
            samples = samples.reshape(-1, 1)
        samples_log = samples_log.reshape(-1, 1)

        return samples, samples_log
