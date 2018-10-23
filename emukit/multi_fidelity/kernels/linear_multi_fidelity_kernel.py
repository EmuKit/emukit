# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Contains kernel for use with Linear Multi Fidelity model
"""

import numpy as np
from GPy.core.parameterization import Param
from GPy.kern.src.kern import CombinationKernel


class LinearMultiFidelityKernel(CombinationKernel):
    def __init__(self, kernels):
        """
        This kernel is used for multi-fidelity problems.

        Args:
            kernels - List of GPy kernels to use for each fidelity from low
                      to high fidelity

        Reference:

        Predicting the output from a complex computer code when fast
        approximations are available. M. C. KENNEDY AND A. O'HAGAN (2000)

        Any number of fidelities are supported.

        Fidelity s is modelled as:
        f_s(x) = p_t * f_t(x) + d_s(x)

        where:
        s is the fidelity
        t is the previous fidelity
        f_s(x) is the function modelling fidelity s
        d_s(x) models the difference between fidelity s-1 and s
        p_t a scaling parameter between fidelity t and s
        """

        self.kernels = kernels
        self.n_fidelities = len(kernels)

        super(LinearMultiFidelityKernel, self).__init__(
            kernels=self.kernels, name='multifidelity', extra_dims=[-1])
        self.scaling_param = Param('scale', np.ones(self.n_fidelities - 1))

        # Link parameters so paramz knows about them
        self.link_parameters(self.scaling_param)

    def K(self, X, X2=None):
        """
        Covariance matrix

        See section 2.5 of Kennedy & O'Hagan paper for details
        """

        if X2 is None:
            X2 = X

        # Build covariance block by block
        K = np.zeros((X.shape[0], X2.shape[0]))
        for i in range(self.n_fidelities):
            for j in range(self.n_fidelities):
                # Looking at block [i, j]
                idx = np.ix_(X[:, -1] == j, X2[:, -1] == i)
                x_this = X[X[:, -1] == j, :]
                x2_this = X2[X2[:, -1] == i, :]
                for k in range(np.min((j + 1, i + 1))):
                    kernel = self.kernels[k].K(x_this, x2_this)
                    scale_1 = np.prod(self.scaling_param[k:i])
                    scale_2 = np.prod(self.scaling_param[k:j])
                    scale = scale_1 * scale_2
                    K[idx] += scale * kernel
        return K

    def Kdiag(self, X):
        """
        Diagonal of covariance matrix

        See section 2.5 of Kennedy & O'Hagan paper for details
        """

        k_diag = np.zeros((X.shape[0]))
        for i in range(self.n_fidelities):
            # Looking at block [i, i]
            idx = np.ix_(X[:, -1] == i)
            for j in range(i + 1):
                kernel = self.kernels[j].Kdiag(X[idx])
                scale = np.prod(self.scaling_param[j:i]**2)
                k_diag[idx] += scale * kernel

        return k_diag

    def gradients_X(self, dL_dK, X, X2=None):
        """
        Gradients of likelihood wrt X. Gradient wrt fidelity index are set to 0
        """

        # Build gradients one block at a time
        dl_dx = np.zeros(X.shape)
        for i in range(self.n_fidelities):
            for j in range(i + 1):
                # Calculate gradients for block [i, j] and [j, i]
                dl_dx += self._calculate_block_matrix_gradients(dL_dK, X, X2,
                                                                i, j)
        return dl_dx

    def _calculate_block_matrix_gradients(self, dL_dK, X, X2, i, j):
        """
        Gradients of likelihood wrt X for block matrix [i, j] and [j, i]
        """

        # Find indices for block [i, j] and block [j, i]
        if X2 is None:
            idx = np.ix_(X[:, -1] == i, X[:, -1] == j)
            idx_t = np.ix_(X[:, -1] == j, X[:, -1] == i)
        else:
            idx = np.ix_(X[:, -1] == i, X2[:, -1] == j)
            idx_t = np.ix_(X[:, -1] == j, X2[:, -1] == i)

        # Take elements of dL_dK corresponding to block matrix
        masked_dl_dk = np.zeros(dL_dK.shape)
        masked_dl_dk[idx] = dL_dK[idx]

        # If block is not on diagonal, make resultant matrix symmetric
        if i != j:
            masked_dl_dk[idx_t] = dL_dK[idx_t]

        block_dl_dx = np.zeros(X.shape)
        for k in range(0, j + 1):
            # Find gradients from sub-kernels + product rule
            scale_1 = np.prod(self.scaling_param[k:i])
            scale_2 = np.prod(self.scaling_param[k:j])
            dl_dk_k = scale_1 * scale_2 * masked_dl_dk
            block_dl_dx += self.kernels[k].gradients_X(dl_dk_k, X, X2)
        return block_dl_dx

    def gradients_X_diag(self, dL_dKdiag, X):
        """
        Gradients of likelihood wrt X. Gradient wrt fidelity index are set to 0
        """

        # Build gradients one fidelity at a time
        dl_dx = np.zeros(X.shape)
        for i in range(self.n_fidelities):
            # Look at all elements at fidelity i
            idx = np.ix_(X[:, -1] == i)

            # Take elements of dL_dKdiag corresponding to this fidelity
            masked_dl_dk = np.zeros(dL_dKdiag.shape)
            masked_dl_dk[idx] = dL_dKdiag[idx]
            for j in range(i + 1):
                scale = np.prod(self.scaling_param[j:i]**2)
                dl_dx += self.kernels[j].gradients_X_diag(masked_dl_dk * scale,
                                                          X)
        return dl_dx

    def update_gradients_diag(self, dL_dKdiag, X):
        """
        Gradients of likelihood wrt model hyper-parameters
        """

        self._update_sub_kernel_gradients_diag(X, dL_dKdiag)

        self._update_scaling_parameter_gradients_diag(X, dL_dKdiag)

    def _update_sub_kernel_gradients_diag(self, X, dL_dKdiag):
        """
        Update gradients of hyper-parameters in sub-kernels
        """

        # Find scaling in front of each kernel term
        dk_dki = []
        for i in range(self.n_fidelities):
            dk_dki.append(np.zeros((X.shape[0])))
            # Look at all X values at fidelity i
            idx = np.ix_(X[:, -1] == i)
            for j in range(i + 1):
                scale = np.prod(self.scaling_param[j:i]**2)
                dk_dki[j][idx] += scale

        # Set gradients in sub-kernels
        for i in range(self.n_fidelities):
            self.kernels[i].update_gradients_diag(dL_dKdiag * dk_dki[i], X)

    def _update_scaling_parameter_gradients_diag(self, X, dL_dKdiag):
        """
        Update gradients of scaling parameters
        """

        # Find K matrices for all sub-kernels
        k_all_fidelities = []
        for i in range(self.n_fidelities):
            k_all_fidelities.append(self.kernels[i].Kdiag(X))

        # Calculate gradients of scaling parameters
        scale_grad = np.zeros(self.n_fidelities - 1)
        for i in range(self.n_fidelities - 1):
            dk_dscale = np.zeros((X.shape[0]))

            # Find gradient for each block matrix at a time
            for j in range(i + 1, self.n_fidelities):
                # block [j, j]
                # Indices for kernel matrices
                idx = np.ix_(X[:, -1] == j)
                for k in range(0, np.min([i + 1, j + 1])):
                    # Find derivative of (coeff * k_l) w.r.t scaling_param[i]
                    # coeff = prod(scaling_param[l:j])^2

                    # This is the scaling term in front of the sub-kernel term
                    tmp1 = np.prod(self.scaling_param[i + 1:j])**2
                    tmp2 = np.prod(self.scaling_param[k:i])**2
                    scale = 2 * tmp1 * tmp2 * self.scaling_param[i]

                    # If square of scaling term appears we need to multiply
                    # the gradient by 2
                    dk_dscale[idx] += scale * k_all_fidelities[k][idx]
            scale_grad[i] = np.sum(dL_dKdiag * dk_dscale)

        self.scaling_param.gradient = scale_grad

    def update_gradients_full(self, dL_dK, X, X2=None):
        """
        Gradients of likelihood wrt model hyper-parameters given dL_dK.
        """

        # X2 is None means X2 == X
        if X2 is None:
            X2 = X

        self._update_sub_kernel_gradients_full(X, X2, dL_dK)

        self._update_scaling_parameter_gradients_full(X, X2, dL_dK)

    def _update_sub_kernel_gradients_full(self, X, X2, dL_dK):
        """
        Update gradients of likelihood wrt sub-kernel hyper-parameters
        """

        # Find the scaling term in front of each sub-kernel
        dk_dki = []
        for i in range(self.n_fidelities):
            dk_dki.append(np.zeros((X.shape[0], X2.shape[0])))
            for j in range(i + 1):
                # Looking at block matrix [i, j]
                idx = np.ix_(X[:, -1] == j, X2[:, -1] == i)
                idx_t = np.ix_(X[:, -1] == i, X2[:, -1] == j)
                for k in range(j + 1):
                    scale_1 = np.prod(self.scaling_param[k:i])
                    scale_2 = np.prod(self.scaling_param[k:j])
                    scale = scale_1 * scale_2
                    dk_dki[k][idx] += scale

                    # Ensure symmetric matrix if this is an off-diagonal block
                    if i != j:
                        dk_dki[k][idx_t] += scale

        # Set gradients in sub-kernels
        for i in range(self.n_fidelities):
            self.kernels[i].update_gradients_full(dL_dK * dk_dki[i], X, X2)

    def _update_scaling_parameter_gradients_full(self, X, X2, dL_dK):
        """
        Update gradients of likelihood wrt scaling parameters
        """
        # First find K matrices for all sub-kernels.
        k_all_fidelities = []
        for i in range(self.n_fidelities):
            k_all_fidelities.append(self.kernels[i].K(X, X2))

        # Calculate gradients of scaling parameters
        scale_grad = np.zeros(self.n_fidelities - 1)
        for i_scale in range(self.n_fidelities - 1):
            scale_grad[i_scale] = self._calculate_d_likelihood_d_scaling_param(
                dL_dK, X, X2, i_scale, k_all_fidelities)

        self.scaling_param.gradient = scale_grad

    def _calculate_d_likelihood_d_scaling_param(
            self, dL_dK, X, X2, i_scaling, k_all_fidelities):
        """
        Calculates gradient of likelihood wrt one scaling parameter

        i_scaling - index of the scale parameter to calculate gradients for
        k_all_fidelities - list of covariance matrices for every fidelity
        """

        # Initialise matrix
        dk_dscale = np.zeros((X.shape[0], X2.shape[0]))

        # Find gradient for each block matrix at a time
        for i_block in range(0, self.n_fidelities):
            for j_block in range(0, self.n_fidelities):
                # Block [i_block, j_block]

                min_fidelity = np.min([i_block, j_block])
                max_fidelity = np.max([i_block, j_block])

                if max_fidelity <= i_scaling:
                    continue

                # Indices for this block
                idx = np.ix_(X[:, -1] == i_block, X2[:, -1] == j_block)

                for i_kern in range(np.min([i_scaling + 1, min_fidelity + 1])):
                    # Find derivative of (coeff*k_l) w.r.t scaling_param[i]
                    # coeff = prod(scaling_param[l:max_fidelity])
                    #         *prod(scaling_param[l:min_fidelity])

                    # Take product of scaling parameters from l to n
                    scale_params_1 = np.prod(
                        self.scaling_param[i_kern:min_fidelity])
                    # Product of scaling parameters from l to j, omitting i
                    scale_params_2 = np.prod(
                        self.scaling_param[i_kern:i_scaling])
                    scale_params_3 = \
                        np.prod(self.scaling_param[i_scaling + 1:max_fidelity])
                    scale = scale_params_1 * scale_params_2 * scale_params_3
                    if (i_scaling < min_fidelity):
                        scale *= 2
                    dk_dscale[idx] += scale * k_all_fidelities[i_kern][idx]
        return np.sum(dL_dK * dk_dscale)
