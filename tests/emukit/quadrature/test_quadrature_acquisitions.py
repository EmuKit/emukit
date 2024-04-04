# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import GPy
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from utils import check_grad

from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.acquisitions import (
    IntegralVarianceReduction,
    MutualInformation,
    SquaredCorrelation,
    UncertaintySampling,
)
from emukit.quadrature.kernels.quadrature_rbf import QuadratureRBFGaussianMeasure, QuadratureRBFLebesgueMeasure
from emukit.quadrature.measures import GaussianMeasure, LebesgueMeasure
from emukit.quadrature.methods import VanillaBayesianQuadrature


@pytest.fixture
def gpy_model():
    rng = np.random.RandomState(42)
    X = rng.rand(5, 2)
    Y = rng.rand(5, 1)
    gpy_kernel = GPy.kern.RBF(input_dim=X.shape[1])
    return GPy.models.GPRegression(X=X, Y=Y, kernel=gpy_kernel)


@pytest.fixture
def model_lebesgue(gpy_model):
    measure = LebesgueMeasure.from_bounds(bounds=gpy_model.X.shape[1] * [(-1, 2)], normalized=False)
    qrbf = QuadratureRBFLebesgueMeasure(RBFGPy(gpy_model.kern), measure)
    basegp = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)
    return VanillaBayesianQuadrature(base_gp=basegp, X=gpy_model.X, Y=gpy_model.Y)


@pytest.fixture
def model_lebesgue_normalized(gpy_model):
    measure = LebesgueMeasure.from_bounds(bounds=gpy_model.X.shape[1] * [(-1, 2)], normalized=True)
    qrbf = QuadratureRBFLebesgueMeasure(RBFGPy(gpy_model.kern), measure)
    basegp = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)
    return VanillaBayesianQuadrature(base_gp=basegp, X=gpy_model.X, Y=gpy_model.Y)


@pytest.fixture
def model_gaussian(gpy_model):
    X, Y = gpy_model.X, gpy_model.Y
    measure = GaussianMeasure(mean=np.arange(gpy_model.X.shape[1]), variance=np.linspace(0.2, 1.5, X.shape[1]))
    qrbf = QuadratureRBFGaussianMeasure(RBFGPy(gpy_model.kern), measure=measure)
    basegp = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)
    return VanillaBayesianQuadrature(base_gp=basegp, X=gpy_model.X, Y=gpy_model.Y)


model_test_list = [
    lazy_fixture("model_gaussian"),
    lazy_fixture("model_lebesgue"),
    lazy_fixture("model_lebesgue_normalized"),
]


@pytest.fixture(params=model_test_list)
def model_test_list_fixture(request):
    return request.param


# === acquisition fixtures start here


@pytest.fixture
def mutual_information(model_test_list_fixture):
    return MutualInformation(model_test_list_fixture)


@pytest.fixture
def squared_correlation(model_test_list_fixture):
    return SquaredCorrelation(model_test_list_fixture)


@pytest.fixture
def integral_variance_reduction(model_test_list_fixture):
    return IntegralVarianceReduction(model_test_list_fixture)


@pytest.fixture
def uncertainty_sampling(model_test_list_fixture):
    return UncertaintySampling(model_test_list_fixture)


acquisitions_test_list = [
    lazy_fixture("mutual_information"),
    lazy_fixture("squared_correlation"),
    lazy_fixture("integral_variance_reduction"),
    lazy_fixture("uncertainty_sampling"),
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
