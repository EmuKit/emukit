# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Tests for multi-fidelity kernels
"""
import GPy
import numpy as np
from GPy.testing.kernel_tests import check_kernel_gradient_functions

import emukit.multi_fidelity


def test_gradidents():
    """
    Use GPy gradient checker to check kernel gradients are implemented
    correctly with many fidelities
    """

    kernels = []
    for i in range(0, 4):
        kernels.append(GPy.kern.RBF(1))
    k = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)

    # Ensure values of scaling parameters and kernel variances aren't 1
    k.scaling_param.value = np.array([0.1, 0.5, 1.5])
    for kern in k.kernels:
        kern.variance.value = np.random.rand(1) + 0.1

    inputs = np.random.rand(20, 2)
    inputs[:5, 1] = 0
    inputs[5:9, 1] = 1
    inputs[9:15, 1] = 2
    inputs[15:] = 3

    inputs_2 = np.random.rand(20, 2)
    inputs_2[:6, 1] = 0
    inputs_2[6:10, 1] = 1
    inputs_2[10:17, 1] = 2
    inputs_2[17:] = 3
    assert check_kernel_gradient_functions(k, X=inputs, X2=inputs_2, verbose=True, fixed_X_dims=-1)


def test_k_full_and_k_diag_are_equivalent():
    """
    Test that kern.K and kern.Kdiag return equivalent results
    """
    kernels = []
    for i in range(0, 2):
        kernels.append(GPy.kern.RBF(1))
    k = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
    inputs = np.random.rand(20, 2)
    inputs[:10, 1] = 1
    inputs[10:, 1] = 0
    assert np.array_equiv(np.diag(k.K(inputs)), k.Kdiag(inputs))


def test_k_x2():
    """
    Test kernel gives expected results when X != X2
    """
    kernels = []
    for i in range(0, 2):
        kernels.append(GPy.kern.RBF(1))
    k = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
    X = np.array([[0, 1], [0, 0]])
    X2 = np.array([[0, 1]])
    cov = k.K(X, X2)
    assert np.all(np.isclose(cov, np.array([[2], [1]])))


def test_k():
    """
    Tests we get expected answer with 3 fidelities
    """
    kernels = []
    for i in range(0, 3):
        kernels.append(GPy.kern.RBF(1))
    k = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)

    k.scaling_param = np.array([0.5, 0.4])
    X = np.array([[0, 0], [0, 1], [0, 2]])
    cov = k.K(X)
    expected_result = np.array([[1, 0.5, 0.2], [0.5, 1.25, 0.5], [0.2, 0.5, 1.2]])
    assert np.all(np.isclose(cov, expected_result))
