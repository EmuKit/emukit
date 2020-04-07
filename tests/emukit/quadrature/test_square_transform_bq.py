# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from math import isclose
import mock
import GPy
import pytest

from emukit.quadrature.interfaces.base_gp import IBaseGaussianProcess
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure
from emukit.quadrature.kernels.quadrature_rbf import QuadratureRBFIsoGaussMeasure
from emukit.quadrature.methods.square_transform_bq import LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature
from emukit.model_wrappers.gpy_quadrature_wrappers import RBFGPy, BaseGaussianProcessGPy


REL_TOL = 1e-5
ABS_TOL = 1e-4


@pytest.fixture
def square_transform_bq():
    X = np.array([[-2, 0], [0, 1], [3, 2]])
    Y = np.array([[-4.1], [-0.1], [-9.1]])
    dim = X.shape[1]
    bound = 0

    gpy_model = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=dim))
    measure = IsotropicGaussianMeasure(mean=np.ones(dim), variance=1.)
    qrbf = QuadratureRBFIsoGaussMeasure(RBFGPy(gpy_model.kern), measure=measure)
    model = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)

    emukit_method = LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature(base_gp=model,
                                                                                X=X, Y=Y,
                                                                                bound=bound)
    return emukit_method

def square_transform_bq_incorrect_bound():
    X = np.array([[-2, 0], [0, 1], [3, 2]])
    Y = np.array([[4], [0], [9]])
    dim = X.shape[1]
    bound = 1  # intentionally incorrect

    gpy_model = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=dim))
    measure = IsotropicGaussianMeasure(mean=np.ones(dim), variance=1.)
    qrbf = QuadratureRBFIsoGaussMeasure(RBFGPy(gpy_model.kern), measure=measure)
    model = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)

    emukit_method = LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature(base_gp=model,
                                                                                X=X, Y=Y,
                                                                                bound=bound)

def test_square_transform_bq_shapes(square_transform_bq):
    X = np.array([[5, 0], [-2, 1]])
    Y = np.array([[-25.1], [-4.1]])

    # integrate
    res = square_transform_bq.integrate()
    assert len(res) == 2
    assert isinstance(res[0], float)
    assert isinstance(res[1], float)

    # transformations
    assert square_transform_bq.transform(Y).shape == Y.shape
    assert square_transform_bq.inverse_transform(Y).shape == Y.shape

    # predictions base
    res = square_transform_bq.predict_base(X)
    assert len(res) == 4
    for i in range(4):
        assert res[i].shape == (X.shape[0], 1)

    # predictions base full covariance
    res = square_transform_bq.predict_base_with_full_covariance(X)
    assert len(res) == 4
    assert res[0].shape == (X.shape[0], 1)
    assert res[1].shape == (X.shape[0], X.shape[0])
    assert res[2].shape == (X.shape[0], 1)
    assert res[3].shape == (X.shape[0], X.shape[0])

    # predictions
    res = square_transform_bq.predict(X)
    assert len(res) == 2
    assert res[0].shape == (X.shape[0], 1)
    assert res[1].shape == (X.shape[0], 1)

    # predictions full covariance
    res = square_transform_bq.predict_with_full_covariance(X)
    assert len(res) == 2
    assert res[0].shape == (X.shape[0], 1)
    assert res[1].shape == (X.shape[0], X.shape[0])

    # predict gradients
    res = square_transform_bq.get_prediction_gradients(X)
    assert len(res) == 2
    assert res[0].shape == (X.shape[0], X.shape[1])
    assert res[1].shape == (X.shape[0], X.shape[1])


def test_square_transform_bq_transformations():
    X = np.random.rand(5, 2)
    Y = np.array([[1], [2], [3], [4], [5]])
    bound = 6

    mock_kern = mock.create_autospec(QuadratureRBFIsoGaussMeasure)
    mock_gp = mock.create_autospec(IBaseGaussianProcess, kern=mock_kern, X=X, Y=Y)

    # no alpha
    method = LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature(base_gp=mock_gp, X=X, Y=Y, bound=bound)

    assert_allclose(method.inverse_transform(Y), np.array([[3.16227766], [2.82842712], [2.44948974],
                                                            [2.], [1.41421356]]))
    assert_allclose(method.transform(Y), np.array([[ 5.5], [ 4. ], [ 1.5], [-2. ], [-6.5]]))
    assert_allclose(method.inverse_transform(method.transform(Y)), Y, rtol=REL_TOL, atol=ABS_TOL)
    assert_allclose(method.transform(method.inverse_transform(Y)), Y, rtol=REL_TOL, atol=ABS_TOL)

    # explicit alpha, not corrected
    method = LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature(base_gp=mock_gp, X=X, Y=Y, bound=bound, alpha=.5)

    assert_allclose(method.inverse_transform(Y), np.array([[3.], [2.64575131], [2.23606798], [1.73205081], [1.]]))
    assert_allclose(method.transform(Y), np.array([[5.], [ 3.5], [ 1. ], [-2.5], [-7. ]]))
    assert_allclose(method.inverse_transform(method.transform(Y)), Y, rtol=REL_TOL, atol=ABS_TOL)
    assert_allclose(method.transform(method.inverse_transform(Y)), Y, rtol=REL_TOL, atol=ABS_TOL)

    # explicit alpha, corrected
    method = LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature(base_gp=mock_gp, X=X, Y=Y, bound=bound, alpha=.6)

    assert_allclose(method.inverse_transform(Y), np.array([[3.], [2.64575131], [2.23606798], [1.73205081], [1.]]))
    assert_allclose(method.transform(Y), np.array([[5.], [ 3.5], [ 1. ], [-2.5], [-7. ]]))
    assert_allclose(method.inverse_transform(method.transform(Y)), Y, rtol=REL_TOL, atol=ABS_TOL)
    assert_allclose(method.transform(method.inverse_transform(Y)), Y, rtol=REL_TOL, atol=ABS_TOL)

    # implied alpha
    Y = - np.array([[1], [2], [3], [4], [5]])
    bound = 0
    method = LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature(base_gp=mock_gp, X=X, Y=Y, bound=bound)

    assert_allclose(method.inverse_transform(Y), np.array([[1.], [1.73205081], [2.23606798], [2.64575131], [3.]]))
    assert_allclose(method.transform(Y), np.array([[ -1. ], [ -2.5], [ -5. ], [ -8.5], [-13. ]]))
    # known effect - sqrt returns the positive root.
    assert_allclose(method.inverse_transform(method.transform(Y)), - Y, rtol=REL_TOL, atol=ABS_TOL)
    assert_allclose(method.transform(method.inverse_transform(Y)), Y, rtol=REL_TOL, atol=ABS_TOL)


def test_square_transform_bq_model():
    X_train = np.random.rand(5, 2)
    Y_train = np.array([[1], [2], [3], [4], [5]])
    Y_train_GP = np.array([[1.1], [2.1], [3.1], [4.1], [5.1]])
    bound = 6

    mock_measure = mock.create_autospec(IsotropicGaussianMeasure)
    mock_kern = mock.create_autospec(QuadratureRBFIsoGaussMeasure, measure=mock_measure)
    model = mock.create_autospec(IBaseGaussianProcess, kern=mock_kern, X=X_train, Y=Y_train_GP)
    method = LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature(base_gp=model, X=X_train, Y=Y_train, bound=bound)

    assert_array_equal(method.X, X_train)
    assert_allclose(method.Y, method.transform(Y_train_GP), rtol=REL_TOL, atol=ABS_TOL)
    assert_array_equal(method.bound, bound)
    assert_array_equal(method.alpha, 0)

    method = LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature(base_gp=model, X=X_train, Y=Y_train,
                                                                         bound=bound, alpha=.5)
    assert_array_equal(method.alpha, .5)

    method = LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature(base_gp=model, X=X_train, Y=Y_train,
                                                                         bound=bound, alpha=.6)
    assert_array_equal(method.alpha, .5)

    method = LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature(base_gp=model, X=X_train, Y=-Y_train,
                                                                         bound=0)
    assert_array_equal(method.alpha, .5)


def test_square_transform_bq_model_invalid_bound():
    with pytest.raises(ValueError):
        X = np.array([[-2], [0], [3]])
        Y = np.array([[4], [0], [9]])
        dim = X.shape[1]
        bound = 1  # intentionally incorrect

        gpy_model = GPy.models.GPRegression(X=X, Y=Y,
                                    kernel=GPy.kern.RBF(input_dim=dim,
                                                        lengthscale=1., variance=1.))
        measure = IsotropicGaussianMeasure(mean=np.ones(dim), variance=1.)

        emukit_qrbf = QuadratureRBFIsoGaussMeasure(rbf_kernel=RBFGPy(gpy_model.kern),
                            measure=measure)

        emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)

        #error
        emukit_method = LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature(base_gp=emukit_model, X=X, Y=Y,
                                                                                    bound=bound)



def test_square_transform_bq_model_alpha_tuning():
    X = np.array([[-2], [1], [3]])
    Y = np.array([[-4.1], [-1.1], [-9.1]])
    dim = X.shape[1]
    bound = 0

    gpy_model = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=dim))
    measure = IsotropicGaussianMeasure(mean=np.ones(dim), variance=1.)
    emukit_qrbf = QuadratureRBFIsoGaussMeasure(rbf_kernel=RBFGPy(gpy_model.kern), measure=measure)
    emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)
    emukit_method = LinSquareTransformRBFIsoGaussMeasureBayesianQuadrature(base_gp=emukit_model,
                                                                    X=X, Y=Y, bound=bound)
    assert emukit_method.alpha == 0.55

    # no change
    X = np.array([[-2], [1], [3], [5]])
    Y = np.array([[-4.1], [-1.1], [-9.1], [-25.1]])

    emukit_method.set_data(X, Y)
    assert emukit_method.alpha == 0.55

    # change
    X = np.array([[-2], [1], [3], [5], [0]])
    Y = np.array([[-4.1], [-1.1], [-9.1], [-25.1], [-.1]])
    emukit_method.set_data(X, Y)
    assert emukit_method.alpha == 0.05


def test_square_transform_bq_integrate(square_transform_bq):
    # to check the integral, we check if it lies in some confidence interval.
    # these intervals were computed as follows: the mean square_transform_bq.predict (first argument) was integrated by
    # simple random sampling with 1e7 samples, and the variance (second argument) with 2*1e4 samples. This was done 100
    # times. The intervals show mean\pm 3 std of the 100 integrals obtained by sampling. There might be a very small
    # chance the true integrals lies outside the specified intervals.
    interval_mean = [-0.6308909408102297, -0.6283025018967014]
    interval_var = [0.10431053077414607, 0.11334763577342227]

    integral_value, integral_variance = square_transform_bq.integrate()
    assert interval_mean[0] < integral_value < interval_mean[1]
    assert interval_var[0] < integral_variance < interval_var[1]


def test_square_transform_bq_gradients(square_transform_bq):
    # boop
    N = 4
    D = 2
    x = np.reshape(np.random.randn(D * N), [N, D])

    # mean
    mean_func = lambda z: square_transform_bq.predict(z)[0]
    mean_grad_func = lambda z: square_transform_bq.get_prediction_gradients(z)[0]
    _check_grad(mean_func, mean_grad_func, x)

    # var
    var_func = lambda z: square_transform_bq.predict(z)[1]
    var_grad_func = lambda z: square_transform_bq.get_prediction_gradients(z)[1]
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
