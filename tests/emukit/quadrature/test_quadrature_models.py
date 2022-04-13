from dataclasses import dataclass
from math import isclose

import GPy
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from pytest_lazyfixture import lazy_fixture
from utils import check_grad


from emukit.quadrature.methods.vanilla_bq import VanillaBayesianQuadrature
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels import QuadratureRBFIsoGaussMeasure, QuadratureRBFLebesgueMeasure
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure
from emukit.quadrature.methods import WSABIL, BoundedBayesianQuadrature


@dataclass
class DataGaussIso:
    D = 2
    measure_mean = np.array([0.2, 1.3])
    measure_var = 2.0
    X = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    Y = np.array([[1], [2], [3]])
    N = 3
    M = 4


def get_gpy_model():
    dat = DataGaussIso()
    return GPy.models.GPRegression(X=dat.X, Y=dat.Y, kernel=GPy.kern.RBF(input_dim=dat.D)), dat


def get_base_gp():
    gpy_model, dat = get_gpy_model()
    measure = IsotropicGaussianMeasure(mean=np.array([0.1, 1.8]), variance=0.8)
    qrbf = QuadratureRBFIsoGaussMeasure(RBFGPy(gpy_model.kern), measure=measure)
    return BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model), dat


def get_base_gp_wrong_kernel():
    gpy_model, dat = get_gpy_model()
    integral_bounds = [(-2.1, 1), (-3, 3)]
    qrbf = QuadratureRBFLebesgueMeasure(RBFGPy(gpy_model.kern), integral_bounds=integral_bounds)
    return BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)


def get_vanilla_bq_model():
    base_gp, dat = get_base_gp()
    return VanillaBayesianQuadrature(base_gp=base_gp, X=dat.X, Y=dat.Y)


def get_bounded_bq_lower():
    base_gp, dat = get_base_gp()
    bound = np.min(dat.Y) - 0.5  # make sure bound is lower than the Y values
    bounded_bq = BoundedBayesianQuadrature(base_gp=base_gp, X=dat.X, Y=dat.Y, lower_bound=bound)
    return bounded_bq, bound


def get_bounded_bq_upper():
    base_gp, dat = get_base_gp()
    bound = np.max(dat.Y) + 0.5  # make sure bound is larger than the Y values
    bounded_bq = BoundedBayesianQuadrature(base_gp=base_gp, X=dat.X, Y=dat.Y, upper_bound=bound)
    return bounded_bq, bound


def get_wsabil_adapt():
    base_gp, dat = get_base_gp()
    wsabil = WSABIL(base_gp=base_gp, X=dat.X, Y=dat.Y, adapt_alpha=True)
    return wsabil, None


def get_wsabil_fixed():
    base_gp, dat = get_base_gp()
    wsabil = WSABIL(base_gp=base_gp, X=dat.X, Y=dat.Y, adapt_alpha=False)
    return wsabil, None


# === fixtures start here
@pytest.fixture
def data():
    return DataGaussIso()


@pytest.fixture
def base_gp():
    return get_base_gp()


@pytest.fixture
def base_gp_wrong_kernel():
    return get_base_gp_wrong_kernel()


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
    assert_array_equal(model.X, data.X)
    assert_array_equal(model.Y, data.Y)


@pytest.mark.parametrize("model", all_models_test_list)
def test_warped_model_shapes(model):
    x = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 4]])
    Y = np.array([[1], [2], [3]])

    N, M = x.shape

    # integrate
    res = model.integrate()
    assert len(res) == 2
    assert isinstance(res[0], float)
    if model in wsabi_test_list:
        assert res[1] is None  # None is returned temporarily until the variance is implemented.
    else:
        assert isinstance(res[1], float)

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
    np.random.seed(0)
    Y = np.random.rand(5, 1)

    # check if warping and inverse warping correctly yield identity.
    assert_allclose(model.inverse_transform(model.transform(Y)), Y)
    assert_allclose(model.transform(model.inverse_transform(Y)), Y)

    # check if warping between base GP and model is consistent.
    Y2 = model.Y
    Y1 = model.base_gp.Y
    assert_allclose(model.transform(Y1), Y2)
    assert_allclose(model.inverse_transform(Y2), Y1)


@pytest.mark.parametrize("model", all_models_test_list)
def test_warped_model_gradient_values(model):

    # gradient of mean
    func = lambda z: model.predict(z)[0][:, 0]
    dfunc = lambda z: model.get_prediction_gradients(z)[0]
    check_grad(func, dfunc, in_shape=(2, 4))

    # gradient of var
    func = lambda z: model.predict(z)[1][:, 0]
    dfunc = lambda z: model.get_prediction_gradients(z)[1]
    check_grad(func, dfunc, in_shape=(2, 4))


@pytest.mark.parametrize(
    "model,interval",
    [
        (vanilla_test_list[0], [0.3378512465045615, 0.34147724727334633]),
        (bounded_test_list[0], ),
        (bounded_test_list[1], ),
        (wsabi_test_list[0], ),
        (wsabi_test_list[1], ),
    ],
)
def test_warped_model_integrate_mean(model, interval):
    res = model.integrate()[0]
    assert interval[0] < res < interval[1]


@pytest.mark.parametrize(
    "model,interval",
    [
        (vanilla_test_list[0], [0.7163111970976184, 0.7404977817143337]),
        (bounded_test_list[0], None),
        (bounded_test_list[1], None),
        (wsabi_test_list[0], None),
        (wsabi_test_list[1], None),
    ],
)
def test_warped_model_integrate_variance(model, interval):
    res = model.integrate()[1]

    if interval is None:
        assert res is None  # None is returned temporarily until the variance is implemented.
    else:
        assert interval[0] < res < interval[1]


# === tests specific to bounded models start here

def test_bounded_bq_correct_bounded_flag(bounded_bq_upper, bounded_bq_lower):
    model, bound = bounded_bq_lower
    assert model.is_lower_bounded
    assert not model._warping.is_inverted

    model, bound = bounded_bq_upper
    assert not model.is_lower_bounded
    assert model._warping.is_inverted


@pytest.mark.parametrize("model", bounded_test_list)
def test_bounded_bq_correct_bound(model):
    # Todo: change this
    model, bound = model
    assert model.bound == bound


def test_bounded_bq_raises(base_gp_wrong_kernel):
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



# to check the integral, we check if it lies in some confidence interval.
# these intervals were computed as follows: the mean vanilla_bq.predict (first argument) was integrated by
# simple random sampling with 1e6 samples, and the variance (second argument) with 5*1e3 samples. This was done 100
# times. The intervals show mean\pm 3 std of the 100 integrals obtained by sampling. There might be a very small
# chance the true integrals lies outside the specified intervals.

# to check the integral, we check if it lies in some confidence interval.
# these intervals were computed as follows: the mean bounded_bq_lower.predict (first argument) was integrated by
# simple random sampling with 1e6 samples. Samples were obtained by sampling from the integration measure.
# The intervals reported are mean\pm 3 std of the 100 integrals obtained by sampling. There might be a very small
# chance that the true integrals lies outside the specified intervals.
