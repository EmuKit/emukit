# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from math import isclose

import GPy
import numpy as np
import pytest

from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.acquisitions import IntegralVarianceReduction, MutualInformation, UncertaintySampling
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure
from emukit.quadrature.kernels.quadrature_rbf import QuadratureRBFIsoGaussMeasure, QuadratureRBFLebesgueMeasure
from emukit.quadrature.methods import VanillaBayesianQuadrature

REL_TOL = 1e-5
ABS_TOL = 1e-4


@pytest.fixture
def model():
    rng = np.random.RandomState(42)
    x_init = rng.rand(5, 2)
    y_init = rng.rand(5, 1)

    gpy_kernel = GPy.kern.RBF(input_dim=x_init.shape[1])
    gpy_model = GPy.models.GPRegression(X=x_init, Y=y_init, kernel=gpy_kernel)
    qrbf = QuadratureRBFLebesgueMeasure(RBFGPy(gpy_kernel), integral_bounds=x_init.shape[1] * [(-3, 3)])
    basegp = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)
    model = VanillaBayesianQuadrature(base_gp=basegp, X=x_init, Y=y_init)
    return model


@pytest.fixture
def model_with_density():
    rng = np.random.RandomState(42)
    x_init = rng.rand(5, 2)
    y_init = rng.rand(5, 1)

    gpy_kernel = GPy.kern.RBF(input_dim=x_init.shape[1])
    gpy_model = GPy.models.GPRegression(X=x_init, Y=y_init, kernel=gpy_kernel)
    measure = IsotropicGaussianMeasure(mean=np.arange(x_init.shape[1]), variance=2.)
    qrbf = QuadratureRBFIsoGaussMeasure(RBFGPy(gpy_kernel), measure=measure)
    basegp = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)
    model = VanillaBayesianQuadrature(base_gp=basegp, X=x_init, Y=y_init)
    return model


def test_mutual_information_shapes(model):
    aq = MutualInformation(model)
    x = np.array([[-1, 1], [0, 0], [-2, 0.1]])

    # value
    res = aq.evaluate(x)
    assert res.shape == (3, 1)

    # gradient
    res = aq.evaluate_with_gradients(x)
    assert res[0].shape == (3, 1)
    assert res[1].shape == (3, 2)


def test_integral_variance_reduction_shapes(model):
    aq = IntegralVarianceReduction(model)
    x = np.array([[-1, 1], [0, 0], [-2, 0.1]])

    # value
    res = aq.evaluate(x)
    assert res.shape == (3, 1)

    # gradient
    res = aq.evaluate_with_gradients(x)
    assert res[0].shape == (3, 1)
    assert res[1].shape == (3, 2)


def test_uncertainty_sampling_shapes(model, model_with_density):
    # test both with Lebesgue measure and with a probability measure because the gradients are computed differently
    x = np.array([[-1, 1], [0, 0], [-2, 0.1]])

    aq = UncertaintySampling(model)
    aq_with_density = UncertaintySampling(model_with_density)

    # value
    res = aq.evaluate(x)
    assert res.shape == (3, 1)

    # gradient
    res = aq.evaluate_with_gradients(x)
    assert res[0].shape == (3, 1)
    assert res[1].shape == (3, 2)

    # value
    res = aq_with_density.evaluate(x)
    assert res.shape == (3, 1)

    # gradient
    res = aq_with_density.evaluate_with_gradients(x)
    assert res[0].shape == (3, 1)
    assert res[1].shape == (3, 2)


def test_mutual_information_gradients(model):
    aq = MutualInformation(model)
    x = np.array([[-2.5, 1.5]])
    _check_grad(aq, x)


def test_integral_variance_reduction_gradients(model):
    aq = IntegralVarianceReduction(model)
    x = np.array([[-2.5, 1.5]])
    _check_grad(aq, x)


def test_uncertainty_sampling_gradients(model, model_with_density):
    aq = UncertaintySampling(model)
    aq_with_density = UncertaintySampling(model_with_density)
    x = np.array([[-2.5, 1.5]])
    _check_grad(aq, x)
    _check_grad(aq_with_density, x)


def _compute_numerical_gradient(aq, x, eps=1e-6):
    f, grad = aq.evaluate_with_gradients(x)
    grad_num = np.zeros(grad.shape)
    for d in range(x.shape[1]):
        x_tmp = x.copy()
        x_tmp[:, d] = x_tmp[:, d] + eps
        f_tmp = aq.evaluate(x_tmp)
        grad_num_d = (f_tmp - f) / eps
        grad_num[:, d] = grad_num_d[:, 0]
    return grad, grad_num


def _check_grad(aq, x):
    grad, grad_num = _compute_numerical_gradient(aq, x)
    isclose_all = 1 - np.array([isclose(grad[i, j], grad_num[i, j], rel_tol=REL_TOL, abs_tol=ABS_TOL)
                                for i in range(grad.shape[0]) for j in range(grad.shape[1])])
    assert isclose_all.sum() == 0
