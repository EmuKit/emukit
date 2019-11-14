# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from numpy.testing import assert_array_equal
from math import isclose
import mock
import GPy
import pytest

from emukit.quadrature.interfaces.base_gp import IBaseGaussianProcess
from emukit.quadrature.methods.vanilla_bq import VanillaBayesianQuadrature
from emukit.quadrature.kernels.bounds import BoxBounds
from emukit.quadrature.kernels.quadrature_kernels import QuadratureKernel
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure
from emukit.core.continuous_parameter import ContinuousParameter
from emukit.model_wrappers.gpy_quadrature_wrappers import RBFGPy, BaseGaussianProcessGPy


REL_TOL = 1e-5
ABS_TOL = 1e-4


@pytest.fixture
def vanilla_bq():
    X = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    Y = np.array([[1], [2], [3]])
    D = X.shape[1]
    integral_bounds = [(-1, 2), (-3, 3)]

    gpy_model = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=D))
    qrbf = QuadratureRBFLebesgueMeasure(RBFGPy(gpy_model.kern), integral_bounds=integral_bounds)
    model = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)
    vanilla_bq = VanillaBayesianQuadrature(base_gp=model, X=X, Y=Y)
    return vanilla_bq


def test_vanilla_bq_shapes(vanilla_bq):
    Y = np.array([[1], [2], [3]])
    x = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 4]])

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

    # predict gradients
    res = vanilla_bq.get_prediction_gradients(x)
    assert len(res) == 2
    assert res[0].shape == (x.shape[0], x.shape[1])
    assert res[1].shape == (x.shape[0], x.shape[1])


def test_vanilla_bq_transformations():
    X = np.random.rand(5, 2)
    Y = np.random.rand(5, 1)

    mock_gp = mock.create_autospec(IBaseGaussianProcess)
    method = VanillaBayesianQuadrature(base_gp=mock_gp, X=X, Y=Y)

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
    mock_bounds = mock.create_autospec(BoxBounds)
    mock_bounds.convert_to_list_of_continuous_parameters.return_value = 2 * [mock_cparam]
    mock_kern = mock.create_autospec(QuadratureKernel, integral_bounds=mock_bounds)
    mock_gp = mock.create_autospec(IBaseGaussianProcess, kern=mock_kern, X=X_train, Y=Y_train)
    method = VanillaBayesianQuadrature(base_gp=mock_gp, X=X_train, Y=Y_train)

    assert_array_equal(method.X, X_train)
    assert_array_equal(method.Y, Y_train)
    assert_array_equal(method.integral_bounds.bounds, mock_bounds.bounds)


def test_vanilla_bq_integrate(vanilla_bq):
    # to check the integral, we check if it lies in some confidence interval.
    # these intervals were computed as follows: the mean vanilla_bq.predict (first argument) was integrated by
    # simple random sampling with 1e6 samples, and the variance (second argument) with 5*1e3 samples. This was done 100
    # times. The intervals show mean\pm 3 std of the 100 integrals obtained by sampling. There might be a very small
    # chance the true integrals lies outside the specified intervals.
    interval_mean = [10.020723475428762, 10.09043533562786]
    interval_var = [41.97715934990283, 46.23549367612568]

    integral_value, integral_variance = vanilla_bq.integrate()
    assert interval_mean[0] < integral_value < interval_mean[1]
    assert interval_var[0] < integral_variance < interval_var[1]


def test_vanilla_bq_gradients(vanilla_bq):
    N = 4
    D = 2
    x = np.reshape(np.random.randn(D * N), [N, D])

    # mean
    mean_func = lambda z: vanilla_bq.predict(z)[0]
    mean_grad_func = lambda z: vanilla_bq.get_prediction_gradients(z)[0]
    _check_grad(mean_func, mean_grad_func, x)

    # var
    var_func = lambda z: vanilla_bq.predict(z)[1]
    var_grad_func = lambda z: vanilla_bq.get_prediction_gradients(z)[1]
    _check_grad(var_func, var_grad_func, x)


def _compute_numerical_gradient(func, grad_func, x, eps=1e-6):
    f = func(x)
    grad = grad_func(x)

    grad_num = np.zeros(grad.shape)
    for d in range(x.shape[1]):
        x_tmp = x.copy()
        x_tmp[:, d] = x_tmp[:, d] + eps
        f_tmp = func(x_tmp)
        grad_num_d = (f_tmp - f) / eps
        grad_num[:, d] = grad_num_d[:, 0]
    return grad, grad_num


def _check_grad(func, grad_func, x):
    grad, grad_num = _compute_numerical_gradient(func, grad_func, x)
    isclose_all = 1 - np.array([isclose(grad[i, j], grad_num[i, j], rel_tol=REL_TOL, abs_tol=ABS_TOL)
                                for i in range(grad.shape[0]) for j in range(grad.shape[1])])
    assert isclose_all.sum() == 0
