# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Contains code for non-linear model multi-fidelity model.

It is based on this paper:

Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling.
P. Perdikaris, M. Raissi, A. Damianou, N. D. Lawrence and G. E. Karniadakis (2017)
http://web.mit.edu/parisp/www/assets/20160751.full.pdf
"""
from typing import Tuple, List, Type

import numpy as np
import GPy

from ...core.interfaces import IModel, IDifferentiable
from ..convert_lists_to_array import convert_y_list_to_array, convert_x_list_to_array


def make_non_linear_kernels(base_kernel_class: Type[GPy.kern.Kern],
                            n_fidelities: int, n_input_dims: int, ARD: bool=False) -> List:
    """
    This function takes a base kernel class and constructs the structured multi-fidelity kernels

    At the first level the kernel is simply:
    .. math
        k_{base}(x, x')

    At subsequent levels the kernels are of the form
    .. math
        k_{base}(x, x')k_{base}(y_{i-1}, y{i-1}') + k_{base}(x, x')

    :param base_kernel_class: GPy class definition of the kernel type to construct the kernels at
    :param n_fidelities: Number of fidelities in the model. A kernel will be returned for each fidelity
    :param n_input_dims: The dimensionality of the input.
    :param ARD: If True, uses different lengthscales for different dimensions. Otherwise the same lengthscale is used
                for all dimensions. Default False.
    :return: A list of kernels with one entry for each fidelity starting from lowest to highest fidelity.
    """

    base_dims_list = list(range(n_input_dims))
    kernels = [base_kernel_class(n_input_dims, active_dims=base_dims_list, ARD=ARD, name='kern_fidelity_1')]
    for i in range(1, n_fidelities):
        fidelity_name = 'fidelity' + str(i + 1)
        interaction_kernel = base_kernel_class(n_input_dims, active_dims=base_dims_list, ARD=ARD,
                                               name='scale_kernel_' + fidelity_name)
        scale_kernel = base_kernel_class(1, active_dims=[n_input_dims], name='previous_fidelity_' + fidelity_name)
        bias_kernel = base_kernel_class(n_input_dims, active_dims=base_dims_list,
                                        ARD=ARD, name='bias_kernel_' + fidelity_name)
        kernels.append(interaction_kernel * scale_kernel + bias_kernel)
    return kernels


class NonLinearMultiFidelityModel(IModel, IDifferentiable):
    """
    Non-linear Model for multiple fidelities. This implementation of the model only handles 1-dimensional outputs.

    The theory implies the training points should be nested such that any point in a higher fidelity exists in all lower
    fidelities, in practice the model will work if this constraint is ignored.
    """

    def __init__(self, X_init: np.ndarray, Y_init: np.ndarray, n_fidelities, kernels: List[GPy.kern.Kern],
                 n_samples=100, verbose=False, optimization_restarts=5) -> None:
        """
        By default the noise at intermediate levels will be fixed to 1e-4.

        :param X_init: Initial X values.
        :param Y_init: Initial Y values.
        :param n_fidelities: Number of fidelities in problem.
        :param kernels: List of kernels for each GP model at each fidelity. The first kernel should take input of
                        dimension d_in and each subsequent kernel should take input of dimension (d_in+1) where d_in is
                        the dimensionality of the features.
        :param n_samples: Number of samples to use to do quasi-Monte-Carlo integration at each fidelity. Default 100
        :param verbose: Whether to output messages during optimization. Defaults to False.
        :param optimization_restarts: Number of random restarts
                                      when optimizing the Gaussian processes' hyper-parameters.
        """

        if not isinstance(X_init, np.ndarray):
            raise TypeError('X_init expected to be a numpy array')

        if not isinstance(Y_init, np.ndarray):
            raise TypeError('Y_init expected to be a numpy array')

        self.verbose = verbose
        self.optimization_restarts = optimization_restarts

        self.n_fidelities = n_fidelities

        # Generate random numbers from standardized gaussian for monte-carlo integration
        self.monte_carlo_rand_numbers = np.random.randn(n_samples)[:, np.newaxis]

        # Make lowest fidelity model
        self.models = []

        self._fidelity_idx = -1

        is_lowest_fidelity = X_init[:, self._fidelity_idx] == 0
        self.models.append(
            GPy.models.GPRegression(X_init[is_lowest_fidelity, :-1], Y_init[is_lowest_fidelity, :], kernels[0]))

        # Make models for fidelities but lowest fidelity
        for i in range(1, self.n_fidelities):
            is_ith_fidelity = X_init[:, self._fidelity_idx] == i
            # Append previous fidelity mean to X
            previous_mean, _ = self._predict_deterministic(X_init[is_ith_fidelity, :-1], i)

            augmented_input = np.concatenate([X_init[is_ith_fidelity, :-1], previous_mean], axis=1)
            self.models.append(GPy.models.GPRegression(augmented_input, Y_init[is_ith_fidelity, :], kernels[i]))

        # Fix noise parameters for all models except top fidelity
        for model in self.models[:-1]:
            model.Gaussian_noise.fix(1e-4)

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Updates training data in the model.

        :param X: New training features.
        :param Y: New training targets.
        """
        is_lowest_fidelity = 0 == X[:, -1]
        X_low_fidelity = X[is_lowest_fidelity, :-1]
        Y_low_fidelity = Y[is_lowest_fidelity, :]
        self.models[0].set_XY(X_low_fidelity, Y_low_fidelity)
        for i in range(1, self.n_fidelities):
            is_this_fidelity = i == X[:, -1]
            X_this_fidelity = X[is_this_fidelity, :-1]
            Y_this_fidelity = Y[is_this_fidelity, :]
            previous_mean, _ = self._predict_deterministic(X_this_fidelity, i)
            augmented_input = np.concatenate([X_this_fidelity, previous_mean], axis=1)
            self.models[i].set_XY(augmented_input, Y_this_fidelity)

    @property
    def X(self):
        """
        :return: input array of size (n_points x n_inputs_dims) across every fidelity in original input domain meaning
                 it excludes inputs to models that come from the output of the previous level
        """
        x_list = [self.models[0].X]
        for model in self.models[1:]:
            x_list.append(model.X[:, :-1])
        return convert_x_list_to_array(x_list)

    @property
    def Y(self):
        """
        :return: output array of size (n_points x n_outputs) across every fidelity level
        """
        return convert_y_list_to_array([model.Y for model in self.models])

    @property
    def n_samples(self):
        return self.monte_carlo_rand_numbers.shape[0]

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance at fidelity given by the last column of X

        Note that the posterior isn't Gaussian and so this function doesn't tell us everything about our posterior
        distribution.

        :param X: Input locations with fidelity index appended.
        :returns: mean and variance of posterior distribution at X.
        """

        fidelity = X[:, self._fidelity_idx]

        # Do prediction 1 test point at a time
        variance = np.zeros((X.shape[0], 1))
        mean = np.zeros((X.shape[0], 1))

        for i in range(X.shape[0]):
            sample_mean, sample_var = self._predict_samples(X[[i], :-1], fidelity[i])
            # Calculate total variance and mean from samples
            variance[i, :] = np.mean(sample_var) + np.var(sample_mean)
            mean[i, :] = np.mean(sample_mean)

        return mean, variance

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance and the gradients of the mean and variance with respect to X.

        :param X: input location.
        :returns: (mean, mean gradient, variance, variance gradient) Gradients will be shape (n_points x (d-1)) because
                  we don't return the gradient with respect to the fidelity index.
        """

        fidelity = X[:, self._fidelity_idx]

        # Initialise vectors
        sample_mean = np.zeros((self.n_samples ** (self.n_fidelities - 1), X.shape[0]))
        d_sample_mean_dx = np.zeros((self.n_samples ** (self.n_fidelities - 1), X.shape[0], X.shape[1] - 1))
        d_sample_var_dx = np.zeros((self.n_samples ** (self.n_fidelities - 1), X.shape[0], X.shape[1] - 1))

        # Iteratively obtain predictions and associated gradients for each input point
        for i in range(X.shape[0]):
            mean, dmean_dx, var, dvar_dx = self._predict_samples_with_gradients(X[[i], :-1], fidelity[i])
            # Assign to outputs
            sample_mean[:, [i]] = mean
            d_sample_mean_dx[:, i, :] = dmean_dx
            d_sample_var_dx[:, i, :] = dvar_dx

        # Calculate means + total variance
        total_mean = np.mean(sample_mean, axis=0, keepdims=True).T
        total_mean_grad = np.mean(d_sample_mean_dx, axis=0)

        # Calculate total variance derivative
        tmp = 2 * np.mean(d_sample_mean_dx * sample_mean[:, :, None], axis=0)
        total_variance_grad = np.mean(d_sample_var_dx, axis=0) + tmp - 2 * total_mean * total_mean_grad

        return total_mean_grad, total_variance_grad

    def _predict_samples(self, X: np.ndarray, fidelity: float):
        """
        Draw samples from model at given fidelity. Returns samples of mean and variance at specified fidelity.
        :param X: Input array without output of previous layer appended.
        :param fidelity: zero based fidelity index.
        :returns sample_mean, sample_variance: mean and variance predictions at input points.
        """

        fidelity = int(fidelity)

        # Predict at first fidelity
        sample_mean, sample_variance = self.models[0].predict(X)

        # Predict at all fidelities up until the one we are interested in
        for i in range(1, fidelity + 1):
            # Draw samples from posterior of previous fidelity
            sample_mean, sample_variance, _ = self._propagate_samples_through_level(X, i, sample_mean, sample_variance)
        return sample_mean, sample_variance

    def _predict_samples_with_gradients(self, X: np.ndarray, fidelity: float):
        """
        Draw samples of mean and variance from model at given fidelity and the gradients of these samples wrt X.

        We calculate the gradients by applying the chain rule as the gradients of each Gaussian process is known wrt
        its inputs.

        :param X: Input array without output of previous layer appended.
        :param fidelity: zero based fidelity index.
        :returns mean, mean gradient, variance, variance gradient: mean and variance predictions at input points.
        """

        fidelity = int(fidelity)

        # Predict at first fidelity
        dsample_mean_dx, dsample_var_dx = self.models[0].predictive_gradients(X)
        dsample_mean_dx = dsample_mean_dx[:, :, 0]
        sample_mean, sample_variance = self.models[0].predict(X)

        for i in range(1, fidelity + 1):
            previous_sample_variance = sample_variance.copy()
            # Predict at all fidelities up until the one we are interested in
            sample_mean, sample_variance, x_augmented = \
                self._propagate_samples_through_level(X, i, sample_mean, sample_variance)
            dsample_mean_dx, dsample_var_dx = \
                self._propagate_samples_through_level_gradient(dsample_mean_dx, dsample_var_dx,
                                                               i, previous_sample_variance, x_augmented)
        return sample_mean, dsample_mean_dx, sample_variance, dsample_var_dx

    def _propagate_samples_through_level(self, X, i_level, sample_mean, sample_variance):
        """
        Sample from the posterior of level i - 1 and propagates these samples through level i.

        :param X: Input array without output of previous layer appended.
        :param i_level: level to push through
        :param sample_mean: mean from previous level
        :param sample_variance: variance from previous level
        """
        # Draw samples from posterior of previous fidelity
        samples = self.monte_carlo_rand_numbers * np.sqrt(sample_variance) + sample_mean.T
        samples = samples.flatten()[:, None]
        # Create inputs for each sample
        x_repeat = np.repeat(X, self.n_samples ** i_level, axis=0)
        # Augment input with mean of previous fidelity
        x_augmented = np.concatenate([x_repeat, samples], axis=1)
        # Predict mean and variance and fidelity i
        sample_mean, sample_variance = self.models[i_level].predict(x_augmented)
        return sample_mean, sample_variance, x_augmented

    def _propagate_samples_through_level_gradient(self, dsample_mean_dx, dsample_var_dx, i_fidelity, sample_variance,
                                                  x_augmented):
        """
        Calculates gradients of sample mean and variance with respect to X when propagated through a level

        :param dsample_mean_dx: Gradients of mean prediction of samples from previous level
        :param dsample_var_dx: Gradients of variance prediction of samples from previous level
        :param i_fidelity: level index
        :param sample_variance: The variance prediction of the samples from the previous level
        :param x_augmented: The X input for this level augmented with the outputs
                            from the previous level as the final column
        """
        # Convert variance derivative to std derivative
        clipped_var = np.clip(sample_variance, 1e-10, np.inf)
        dsample_std_dx = dsample_var_dx / (2 * np.sqrt(clipped_var))
        # Calculate gradients of samples wrt x
        # This calculates a (n_samples**(i-1), n_samples, n_dims) matrix
        tmp = self.monte_carlo_rand_numbers[:, np.newaxis, :] * dsample_std_dx[:, np.newaxis, :]
        dsamples_dx = dsample_mean_dx[np.newaxis, :, :] + tmp
        dsamples_dx_reshaped = np.reshape(dsamples_dx, (self.n_samples ** i_fidelity, dsample_std_dx.shape[1]))

        # Get partial derivatives of mean and variance with respect to
        # both X and output of previous fidelity
        dmean_dx, dvar_dx = self.models[i_fidelity].predictive_gradients(x_augmented)
        dmean_dx = dmean_dx[:, :, 0]

        # Combine partial derivatives to get full derivative wrt X
        dsample_mean_dx = dmean_dx[:, :-1] + dmean_dx[:, [-1]] * dsamples_dx_reshaped
        dsample_var_dx = dvar_dx[:, :-1] + dvar_dx[:, [-1]] * dsamples_dx_reshaped
        return dsample_mean_dx, dsample_var_dx

    def optimize(self) -> None:
        """
        Optimize the full model
        """

        # Optimize the first model
        self.models[0].optimize_restarts(self.optimization_restarts, verbose=self.verbose)

        # Optimize all models for all fidelities but lowest fidelity
        for i in range(1, self.n_fidelities):
            # Set new X values because previous model has changed
            previous_mean, _ = self.models[i - 1].predict(self.models[i].X)
            augmented_input = np.concatenate([self.models[i].X[:, :-1], previous_mean], axis=1)
            self.models[i].set_X(augmented_input)

            # Optimize parameters
            self.models[i].optimize_restarts(self.optimization_restarts, verbose=self.verbose)

    def get_f_minimum(self) -> np.ndarray:
        """
        Get the minimum of the top fidelity model.
        """
        return np.min(self.models[-1].Y)

    def _predict_deterministic(self, X, fidelity):
        """
        This is a helper function when predicting at points that are in the training set. It is more efficient than
        sampling and is useful when constructing the model.
        """
        # Predict at first fidelity
        mean, variance = self.models[0].predict(X)
        for i in range(1, fidelity):
            # Push samples through this fidelity model
            augmented_input = np.concatenate([X, mean], axis=1)
            mean, variance = self.models[i].predict(augmented_input)
        return mean, variance
