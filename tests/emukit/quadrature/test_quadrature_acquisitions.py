# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import GPy
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from utils import check_grad

from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.acquisitions import IntegralVarianceReduction, MutualInformation, UncertaintySampling
from emukit.quadrature.kernels.quadrature_rbf import QuadratureRBFGaussianMeasure, QuadratureRBFLebesgueMeasure
from emukit.quadrature.measures import GaussianMeasure, LebesgueMeasure
from emukit.quadrature.methods import VanillaBayesianQuadrature


@pytest.fixture
def gpy_model():
    rng = np.random.RandomState(42)
    x_init = rng.rand(5, 2)
    y_init = rng.rand(5, 1)
    gpy_kernel = GPy.kern.RBF(input_dim=x_init.shape[1])
    gpy_model = GPy.models.GPRegression(X=x_init, Y=y_init, kernel=gpy_kernel)
    return gpy_model


@pytest.fixture
def model(gpy_model):
    x_init, y_init = gpy_model.X, gpy_model.Y
    measure = LebesgueMeasure.from_bounds(bounds=x_init.shape[1] * [(-3, 3)])
    qrbf = QuadratureRBFLebesgueMeasure(RBFGPy(gpy_model.kern), measure)
    basegp = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)
    return VanillaBayesianQuadrature(base_gp=basegp, X=x_init, Y=y_init)


@pytest.fixture
def model_with_density(gpy_model):
    x_init, y_init = gpy_model.X, gpy_model.Y
    measure = GaussianMeasure(mean=np.arange(x_init.shape[1]), variance=2.0)
    qrbf = QuadratureRBFGaussianMeasure(RBFGPy(gpy_model.kern), measure=measure)
    basegp = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)
    return VanillaBayesianQuadrature(base_gp=basegp, X=x_init, Y=y_init)


# === acquisition fixtures start here


@pytest.fixture
def mutual_information(model):
    return MutualInformation(model)


@pytest.fixture
def integral_variance_reduction(model):
    return IntegralVarianceReduction(model)


@pytest.fixture
def uncertainty_sampling(model):
    return UncertaintySampling(model)


@pytest.fixture
def uncertainty_sampling_density(model_with_density):
    return UncertaintySampling(model_with_density)


acquisitions_test_list = [
    lazy_fixture("mutual_information"),
    lazy_fixture("integral_variance_reduction"),
    lazy_fixture("uncertainty_sampling"),
    lazy_fixture("uncertainty_sampling_density"),
]


@pytest.mark.parametrize("aq", acquisitions_test_list)
def test_quadrature_acquisition_shapes(aq):
    x = np.array([[-1, 1], [0, 0], [-2, 0.1]])

    # value
    res = aq.evaluate(x)
    assert res.shape == (3, 1)

    # gradient
    res = aq.evaluate_with_gradients(x)
    assert res[0].shape == (3, 1)
    assert res[1].shape == (3, 2)


@pytest.mark.parametrize("aq", acquisitions_test_list)
def test_quadrature_acquisition_gradient_values(aq):
    func = lambda x: aq.evaluate(x)[:, 0]
    dfunc = lambda x: aq.evaluate_with_gradients(x)[1].T
    check_grad(func, dfunc, in_shape=(3, 2), bounds=aq.model.X.shape[1] * [(-3, 3)])
