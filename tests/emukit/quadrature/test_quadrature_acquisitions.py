# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import GPy
import numpy as np
import pytest
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
    "model_gaussian",
    "model_lebesgue",
    "model_lebesgue_normalized",
]

# === acquisition fixtures start here


@pytest.fixture
def mutual_information():
    return lambda model: MutualInformation(model)


@pytest.fixture
def squared_correlation():
    return lambda model: SquaredCorrelation(model)


@pytest.fixture
def integral_variance_reduction():
    return lambda model: IntegralVarianceReduction(model)


@pytest.fixture
def uncertainty_sampling():
    return lambda model: UncertaintySampling(model)


acquisition_test_list = [
    "mutual_information",
    "squared_correlation",
    "integral_variance_reduction",
    "uncertainty_sampling",
]


@pytest.mark.parametrize("model_name", model_test_list)
@pytest.mark.parametrize("aq_name", acquisition_test_list)
def test_quadrature_acquisition_shapes(model_name, aq_name, request):
    model = request.getfixturevalue(model_name)
    aq_factory = request.getfixturevalue(aq_name)
    aq = aq_factory(model)

    x = np.array([[-1, 1], [0, 0], [-2, 0.1]])

    # value
    res = aq.evaluate(x)
    assert res.shape == (3, 1)

    # gradient
    res = aq.evaluate_with_gradients(x)
    assert res[0].shape == (3, 1)
    assert res[1].shape == (3, 2)


@pytest.mark.parametrize("model_name", model_test_list)
@pytest.mark.parametrize("aq_name", acquisition_test_list)
def test_quadrature_acquisition_gradient_values(model_name, aq_name, request):
    model = request.getfixturevalue(model_name)
    aq_factory = request.getfixturevalue(aq_name)
    aq = aq_factory(model)

    func = lambda x: aq.evaluate(x)[:, 0]
    dfunc = lambda x: aq.evaluate_with_gradients(x)[1].T
    check_grad(func, dfunc, in_shape=(3, 2), bounds=aq.model.X.shape[1] * [(-3, 3)])
