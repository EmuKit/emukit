# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from numpy.testing import assert_array_equal
import mock
import GPy
import pytest

from emukit.quadrature.interfaces.base_gp import IBaseGaussianProcess
from emukit.quadrature.methods.vanilla_bq import VanillaBayesianQuadrature
from emukit.quadrature.kernels.bounds import BoxBounds
from emukit.quadrature.kernels.quadrature_kernels import QuadratureKernel
from emukit.quadrature.kernels import QuadratureRBF
from emukit.core.continuous_parameter import ContinuousParameter
from emukit.model_wrappers.gpy_quadrature_wrappers import RBFGPy, BaseGaussianProcessGPy


@pytest.fixture
def vanilla_bq():
    X = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    Y = np.array([[1], [2], [3]])
    D = X.shape[1]
    integral_bounds = D * [(-3, 3)]

    gpy_model = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=D))
    qrbf = QuadratureRBF(RBFGPy(gpy_model.kern), integral_bounds=integral_bounds)
    model = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)
    vanilla_bq = VanillaBayesianQuadrature(base_gp=model)
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


def test_vanilla_bq_set_new_bounds(vanilla_bq):
    new_bounds = 2 * [(-2, 2)]
    new_lower = np.array([[-2, -2]])
    new_upper = np.array([[2, 2]])
    vanilla_bq.integral_bounds = new_bounds

    res = vanilla_bq.integral_bounds.get_lower_and_upper_bounds()
    assert_array_equal(res[0], new_lower)
    assert_array_equal(res[1], new_upper)

    res = vanilla_bq.integral_bounds.bounds
    assert all([res_i == nb_i for res_i, nb_i in zip(res, new_bounds)])


def test_vanilla_bq_set_wrong_bounds(vanilla_bq):
    # wrong dimensionality
    wrong_dim = vanilla_bq.integral_bounds.dim + 1
    wrong_bounds = wrong_dim * [(-2, 2)]
    with pytest.raises(ValueError):
        vanilla_bq.integral_bounds = wrong_bounds

    # empty bounds
    wrong_bounds = []
    with pytest.raises(ValueError):
        vanilla_bq.integral_bounds = wrong_bounds

    # wrong bound values
    wrong_bounds = 3 * [(-2, -3)]
    with pytest.raises(ValueError):
        vanilla_bq.integral_bounds = wrong_bounds


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


def test_vanilla_bq_model(vanilla_bq):
    X_train = np.random.rand(5, 2)
    Y_train = np.random.rand(5, 1)

    mock_cparam = mock.create_autospec(ContinuousParameter)
    mock_bounds = mock.create_autospec(BoxBounds)
    mock_bounds.convert_to_list_of_continuous_parameters.return_value = 2 * [mock_cparam]
    mock_kern = mock.create_autospec(QuadratureKernel, integral_bounds=mock_bounds)
    mock_gp = mock.create_autospec(IBaseGaussianProcess, kern=mock_kern, X=X_train, Y=Y_train)
    method = VanillaBayesianQuadrature(base_gp=mock_gp)

    assert_array_equal(method.X, X_train)
    assert_array_equal(method.Y, Y_train)
    assert_array_equal(method.integral_bounds.bounds, mock_bounds.bounds)
    # we assert that the integral bounds are the same in the method and the quadrature kernel, since all integration
    # happens in the kernel. The test is restrictive but easier to do for the init than behavioral test. There are some
    # behavioral tests for setting new bounds in the vanilla_bq method and the bounds itself. Setting the integral
    # bounds should be done with care.
    assert method.integral_bounds == mock_bounds
    assert method.integral_parameters == 2 * [mock_cparam]
