# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import mock
from numpy.testing import assert_array_equal

from emukit.quadrature.interfaces.base_gp import IBaseGaussianProcess
from emukit.quadrature.methods.vanilla_bq import VanillaBayesianQuadrature
from emukit.quadrature.kernels.integral_bounds import IntegralBounds
from emukit.quadrature.kernels.quadrature_kernels import QuadratureKernel
from emukit.core.continuous_parameter import ContinuousParameter


def test_vanilla_bq_transformations():
    Y = np.random.rand(5, 1)

    mock_gp = mock.create_autospec(IBaseGaussianProcess)
    method = VanillaBayesianQuadrature(base_gp=mock_gp)

    # we can use equal comparison here because vanilla bq uses the identity transform. For non-trivial transforms
    # with numerical errors use a closeness test instead.
    assert_array_equal(method.inverse_transform(Y), Y)
    assert_array_equal(method.transform(Y), Y)
    assert_array_equal(method.inverse_transform(method.transform(Y)), Y)
    assert_array_equal(method.transform(method.inverse_transform(Y)), Y)


def test_vanilla_bq_model():
    X_train = np.random.rand(5, 2)
    Y_train = np.random.rand(5, 1)

    mock_cparam = mock.create_autospec(ContinuousParameter)
    mock_bounds = mock.create_autospec(IntegralBounds)
    mock_bounds.convert_to_list_of_continuous_parameters.return_value = 2 * [mock_cparam]
    mock_kern = mock.create_autospec(QuadratureKernel, integral_bounds=mock_bounds)
    mock_gp = mock.create_autospec(IBaseGaussianProcess, kern=mock_kern, X=X_train, Y=Y_train)
    method = VanillaBayesianQuadrature(base_gp=mock_gp)

    assert_array_equal(method.X, X_train)
    assert_array_equal(method.Y, Y_train)
    # we assert this to make sure that integral bounds in the kernel match, since that is where the integration happens.
    # the test is restrictive but easier to do than behavioral test when integral bounds are changed.
    assert method.integral_bounds == mock_bounds
    assert method.integral_parameters == 2 * [mock_cparam]
