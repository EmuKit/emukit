# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from numpy.testing import assert_array_equal
import mock
import GPy

from emukit.quadrature.interfaces.base_gp import IBaseGaussianProcess
from emukit.quadrature.methods.vanilla_bq import VanillaBayesianQuadrature
from emukit.quadrature.kernels.integral_bounds import IntegralBounds
from emukit.quadrature.kernels.quadrature_kernels import QuadratureKernel
from emukit.quadrature.kernels import QuadratureRBF
from emukit.core.continuous_parameter import ContinuousParameter
from emukit.model_wrappers.gpy_quadrature_wrappers import RBFGPy, BaseGaussianProcessGPy


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


def test_vanilla_bq_shapes():
    X = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    Y = np.array([[1], [2], [3]])
    x = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 4]])
    D = X.shape[1]

    # integral bounds
    lb = -3
    ub = 3

    gpy_model = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=D))
    emukit_qrbf = QuadratureRBF(RBFGPy(gpy_model.kern), integral_bounds=D * [(lb, ub)])
    emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)
    vanilla_bq = VanillaBayesianQuadrature(base_gp=emukit_model)

    # integrate
    res = vanilla_bq.integrate()
    assert len(res) == 2
    assert isinstance(res[0], float)
    assert isinstance(res[1], float)

    # transformations
    assert vanilla_bq.transform(Y).shape == Y.shape
    assert vanilla_bq.inverse_transform(Y).shape == Y.shape

    # predictions base
    res = vanilla_bq.predict_base(x)
    assert len(res) == 4
    for i in range(4):
        assert res[i].shape == (x.shape[0], 1)

    # predictions base full covariance
    res = vanilla_bq.predict_base_with_full_covariance(x)
    assert len(res) == 4
    assert res[0].shape == (x.shape[0], 1)
    assert res[1].shape == (x.shape[0], x.shape[0])
    assert res[2].shape == (x.shape[0], 1)
    assert res[3].shape == (x.shape[0], x.shape[0])

    # predictions
    res = vanilla_bq.predict(x)
    assert len(res) == 2
    assert res[0].shape == (x.shape[0], 1)
    assert res[1].shape == (x.shape[0], 1)

    # predictions full covariance
    res = vanilla_bq.predict_with_full_covariance(x)
    assert len(res) == 2
    assert res[0].shape == (x.shape[0], 1)
    assert res[1].shape == (x.shape[0], x.shape[0])
