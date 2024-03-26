# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from math import isclose

import GPy
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytest_lazyfixture import lazy_fixture
from utils import check_grad

from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels import QuadratureRBFGaussianMeasure, QuadratureRBFLebesgueMeasure
from emukit.quadrature.measures import GaussianMeasure, LebesgueMeasure
from emukit.quadrature.methods import WSABIL, BoundedBayesianQuadrature
from emukit.quadrature.methods.vanilla_bq import VanillaBayesianQuadrature


@dataclass
class DataGaussianSpread:
    D = 2
    measure_mean = np.array([0.2, 1.3])
    measure_var = 2.0
    X = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    Y = np.array([[1], [2], [3]])

    # for bounded methods
    bound_lower = np.min(Y) - 0.5  # make sure bound is lower than the Y values
    bound_upper = np.max(Y) + 0.5  # make sure bound is larger than the Y values
    dat_bounds = [(m - 2 * np.sqrt(2), m + 2 * np.sqrt(2)) for m in measure_mean]


def get_gpy_model():
    dat = DataGaussianSpread()
    gpy_kern = GPy.kern.RBF(input_dim=dat.D)
    return GPy.models.GPRegression(X=dat.X, Y=dat.Y, kernel=gpy_kern), dat


def get_base_gp():
    gpy_model, dat = get_gpy_model()
    measure = GaussianMeasure(mean=dat.measure_mean, variance=dat.measure_var)
    qrbf = QuadratureRBFGaussianMeasure(RBFGPy(gpy_model.kern), measure=measure)
    return BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model), dat


def get_vanilla_bq_model():
    base_gp, dat = get_base_gp()
    return VanillaBayesianQuadrature(base_gp=base_gp, X=dat.X, Y=dat.Y)


def get_bounded_bq_lower():
    base_gp, dat = get_base_gp()
    return BoundedBayesianQuadrature(base_gp=base_gp, X=dat.X, Y=dat.Y, lower_bound=dat.bound_lower)


def get_bounded_bq_upper():
    base_gp, dat = get_base_gp()
    return BoundedBayesianQuadrature(base_gp=base_gp, X=dat.X, Y=dat.Y, upper_bound=dat.bound_upper)


def get_wsabil_adapt():
    base_gp, dat = get_base_gp()
    return WSABIL(base_gp=base_gp, X=dat.X, Y=dat.Y, adapt_alpha=True)


def get_wsabil_fixed():
    base_gp, dat = get_base_gp()
    wsabil = WSABIL(base_gp=base_gp, X=dat.X, Y=dat.Y, adapt_alpha=False)
    return wsabil


# === fixtures start here
@pytest.fixture
def data():
    return DataGaussianSpread()


@pytest.fixture
def gpy_model():
    return get_gpy_model()


@pytest.fixture
def base_gp():
    return get_base_gp()


@pytest.fixture
def vanilla_bq():
    return get_vanilla_bq_model()


@pytest.fixture
def bounded_bq_lower():
    return get_bounded_bq_lower()


@pytest.fixture
def bounded_bq_upper():
    return get_bounded_bq_upper()


@pytest.fixture
def wsabil_adapt():
    return get_wsabil_adapt()


@pytest.fixture
def wsabil_fixed():
    return get_wsabil_fixed()


vanilla_test_list = [
    lazy_fixture("vanilla_bq"),
]

bounded_test_list = [
    lazy_fixture("bounded_bq_lower"),
    lazy_fixture("bounded_bq_upper"),
]

wsabi_test_list = [
    lazy_fixture("wsabil_adapt"),
    lazy_fixture("wsabil_fixed"),
]

all_models_test_list = vanilla_test_list + bounded_test_list + wsabi_test_list


# === tests shared by all warped models start here


@pytest.mark.parametrize("model", all_models_test_list)
def test_warped_model_data(model, data):
    ABS_TOL = 1e-5
    REL_TOL = 1e-6
    assert_allclose(model.X, data.X, rtol=REL_TOL, atol=ABS_TOL)
    assert_allclose(model.Y, data.Y, rtol=REL_TOL, atol=ABS_TOL)


@pytest.mark.parametrize("model", all_models_test_list)
def test_warped_model_shapes(model):
    x = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 4]])
    Y = np.array([[1], [2], [3]])

    N, M = x.shape

    # integrate
    res = model.integrate()
    assert len(res) == 2
    assert isinstance(res[0], float)

    if isinstance(model, VanillaBayesianQuadrature):
        assert isinstance(res[1], float)
    else:
        assert res[1] is None  # None is returned temporarily until the variance is implemented.

    # transformations
    assert model.transform(Y).shape == Y.shape
    assert model.inverse_transform(Y).shape == Y.shape

    # predictions base
    res = model.predict_base(x)
    assert len(res) == 4
    for i in range(4):
        assert res[i].shape == (N, 1)

    # predictions base full covariance
    res = model.predict_base_with_full_covariance(x)
    assert len(res) == 4
    assert res[0].shape == (N, 1)
    assert res[1].shape == (N, N)
    assert res[2].shape == (N, 1)
    assert res[3].shape == (N, N)

    # predictions
    res = model.predict(x)
    assert len(res) == 2
    assert res[0].shape == (N, 1)
    assert res[1].shape == (N, 1)

    # predictions full covariance
    res = model.predict_with_full_covariance(x)
    assert len(res) == 2
    assert res[0].shape == (N, 1)
    assert res[1].shape == (N, N)

    # predict gradients
    res = model.get_prediction_gradients(x)
    assert len(res) == 2
    assert res[0].shape == (N, M)
    assert res[1].shape == (N, M)


@pytest.mark.parametrize("model", all_models_test_list)
def test_warped_model_transforms(model):
    Y = np.array([[1], [2], [3]])
    ABS_TOL = 1e-5
    REL_TOL = 1e-6

    # check if warping and inverse warping correctly yield identity.
    assert_allclose(model.inverse_transform(model.transform(Y)), Y, rtol=REL_TOL, atol=ABS_TOL)
    assert_allclose(model.transform(model.inverse_transform(Y)), Y, rtol=REL_TOL, atol=ABS_TOL)

    # check if warping between base GP and model is consistent.
    Y2 = model.Y
    Y1 = model.base_gp.Y
    assert_allclose(model.transform(Y1), Y2, rtol=REL_TOL, atol=ABS_TOL)
    assert_allclose(model.inverse_transform(Y2), Y1, rtol=REL_TOL, atol=ABS_TOL)


@pytest.mark.parametrize("model", all_models_test_list)
def test_warped_model_gradient_values(model, data):
    # gradient of mean
    func = lambda z: model.predict(z)[0][:, 0]
    dfunc = lambda z: model.get_prediction_gradients(z)[0].T
    check_grad(func, dfunc, in_shape=(5, data.D), bounds=data.dat_bounds)

    # gradient of var
    func = lambda z: model.predict(z)[1][:, 0]
    dfunc = lambda z: model.get_prediction_gradients(z)[1].T
    check_grad(func, dfunc, in_shape=(5, data.D), bounds=data.dat_bounds)


@pytest.mark.parametrize(
    "model,interval",
    [
        (vanilla_test_list[0], [0.5956279650321574, 0.6000811779371775]),
        (bounded_test_list[0], [0.8383067891004425, 0.8417905366769567]),
        (bounded_test_list[1], [2.977651803340788, 2.981939540780773]),
        (wsabi_test_list[0], [1.0571955335349208, 1.0601420159245922]),
        (wsabi_test_list[1], [0.47610638476406725, 0.48068140048609603]),
    ],
)
def test_warped_model_integrate_mean(model, interval):
    # Both outputs of the model.intgerate() method are analytic integrals.
    # To test their values we check if they lie in the confidence interva of an MC estimator.
    # These intervals were computed as follows: the mean model.predict (first argument) was integrated by
    # simple random sampling with 1e6 samples, and the variance (second argument) with 5*1e3 samples. This was done 100
    # times. The intervals show mean\pm 3 std of the 100 integrals obtained by sampling. There might be a very small
    # chance that the true integrals lies outside the specified intervals.
    # See file "ground_truth_integrals_methods.py" for details.
    res = model.integrate()[0]
    assert interval[0] < res < interval[1]


@pytest.mark.parametrize(
    "model,interval",
    [
        (vanilla_test_list[0], [0.09859906877945852, 0.11181285735935843]),
        (bounded_test_list[0], None),
        (bounded_test_list[1], None),
        (wsabi_test_list[0], None),
        (wsabi_test_list[1], None),
    ],
)
def test_warped_model_integrate_variance(model, interval):
    # See test_warped_model_integrate_mean on how the intervals were computed
    res = model.integrate()[1]

    if interval is None:
        assert res is None  # None is returned temporarily until the variance is implemented.
    else:
        assert interval[0] < res < interval[1]


# === tests specific to bounded models start here


def test_bounded_bq_correct_bounded_flag(bounded_bq_upper, bounded_bq_lower):
    assert bounded_bq_lower.is_lower_bounded
    assert not bounded_bq_lower._warping.is_inverted

    assert not bounded_bq_upper.is_lower_bounded
    assert bounded_bq_upper._warping.is_inverted


def test_bounded_bq_correct_bound(data, bounded_bq_lower, bounded_bq_upper):
    assert bounded_bq_lower.bound == data.bound_lower
    assert bounded_bq_upper.bound == data.bound_upper


def test_bounded_bq_raises(gpy_model):
    gpy_model, _ = gpy_model
    measure = LebesgueMeasure.from_bounds(gpy_model.X.shape[1] * [(0, 1)], normalized=False)
    qrbf = QuadratureRBFLebesgueMeasure(RBFGPy(gpy_model.kern), measure=measure)
    base_gp_wrong_kernel = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)

    # wrong kernel embedding
    with pytest.raises(ValueError):
        BoundedBayesianQuadrature(
            base_gp=base_gp_wrong_kernel,
            X=base_gp_wrong_kernel.X,
            Y=base_gp_wrong_kernel.Y,
            lower_bound=np.min(base_gp_wrong_kernel.Y) - 0.5,
        )

    # both upper and lower bound are given
    with pytest.raises(ValueError):
        BoundedBayesianQuadrature(
            base_gp=base_gp_wrong_kernel,
            X=base_gp_wrong_kernel.X,
            Y=base_gp_wrong_kernel.Y,
            lower_bound=np.min(base_gp_wrong_kernel.Y) - 0.5,
            upper_bound=np.min(base_gp_wrong_kernel.Y) - 0.5,
        )

    # no bound is given
    with pytest.raises(ValueError):
        BoundedBayesianQuadrature(base_gp=base_gp_wrong_kernel, X=base_gp_wrong_kernel.X, Y=base_gp_wrong_kernel.Y)


# === tests specific to wsabi models start here


def test_wsabi_alpha_adaptation(wsabil_adapt, wsabil_fixed):
    X_new = np.array([[1.1, 1.2], [-1, 1], [0, 0], [-2, 0.1]])
    Y_new = np.array([[0.8], [1], [2], [3]])  # lowest value is 0.8

    # check if alpha is adapted correctly
    model = wsabil_adapt
    model.set_data(X_new, Y_new)
    assert model.adapt_alpha
    assert isclose(model.bound, 0.64)  # 0.8 * min(Y_new)

    # check if alpha stays fixed
    model = wsabil_fixed
    old_alpha = model.bound
    assert not model.adapt_alpha
    assert model.bound == old_alpha
