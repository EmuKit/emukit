import GPy
import pytest
import numpy as np
from numpy.testing import assert_allclose
from math import isclose
from pytest_lazyfixture import lazy_fixture

from emukit.quadrature.methods import BoundedBayesianQuadratureModel, WSABIL
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure, QuadratureRBFIsoGaussMeasure
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure
from emukit.model_wrappers.gpy_quadrature_wrappers import RBFGPy, BaseGaussianProcessGPy

REL_TOL = 1e-5
ABS_TOL = 1e-4


@pytest.fixture
def gpy_model():
    X = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    Y = np.array([[1], [2], [3]])
    return GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=X.shape[1]))


@pytest.fixture
def base_gp_wrong_kernel(gpy_model):
    integral_bounds = [(-1, 2), (-3, 3)]
    qrbf = QuadratureRBFLebesgueMeasure(RBFGPy(gpy_model.kern), integral_bounds=integral_bounds)
    return BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)


@pytest.fixture
def base_gp(gpy_model):
    measure = IsotropicGaussianMeasure(mean=np.array([0.1, 1.8]), variance=0.8)
    qrbf = QuadratureRBFIsoGaussMeasure(RBFGPy(gpy_model.kern), measure=measure)
    return BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)


@pytest.fixture
def bounded_bq_lower(base_gp):
    bound = np.min(base_gp.Y) - 0.5  # make sure bound is lower than the Y values
    bounded_bq = BoundedBayesianQuadratureModel(base_gp=base_gp, X=base_gp.X, Y=base_gp.Y, bound=bound, is_lower_bounded=True)
    return bounded_bq, bound


@pytest.fixture
def bounded_bq_upper(base_gp):
    bound = np.max(base_gp.Y) + 0.5  # make sure bound is larger than the Y values
    bounded_bq = BoundedBayesianQuadratureModel(base_gp=base_gp, X=base_gp.X, Y=base_gp.Y, bound=bound, is_lower_bounded=False)
    return bounded_bq, bound


@pytest.fixture
def wsabil_adapt(base_gp):
    wsabil = WSABIL(base_gp=base_gp, X=base_gp.X, Y=base_gp.Y, adapt_alpha=True)
    return wsabil, None


@pytest.fixture
def wsabil_fixed(base_gp):
    wsabil = WSABIL(base_gp=base_gp, X=base_gp.X, Y=base_gp.Y, adapt_alpha=False)
    return wsabil, None


models_test_list = [lazy_fixture("bounded_bq_lower"), lazy_fixture("bounded_bq_upper")]
wsabi_test_list = [lazy_fixture("wsabil_adapt"), lazy_fixture("wsabil_fixed")]


# === tests specific to bounded BQ
@pytest.mark.parametrize('bounded_bq', models_test_list)
def test_bounded_bq_correct_bound(bounded_bq):
    model, bound = bounded_bq
    assert model.bound == bound


def test_bounded_bq_raises_exception(base_gp_wrong_kernel):
    # wrong kernel embedding
    with pytest.raises(ValueError):
        BoundedBayesianQuadratureModel(base_gp=base_gp_wrong_kernel, X=base_gp_wrong_kernel.X, Y=base_gp_wrong_kernel.Y,
                                       bound=np.min(base_gp_wrong_kernel.Y) - 0.5, is_lower_bounded=True)


# === tests specific to WSABI
def test_wsabi_alpha_adaptation(wsabil_adapt, wsabil_fixed):
    X_new = np.array([[1.1, 1.2], [-1, 1], [0, 0], [-2, 0.1]])
    Y_new = np.array([[0.8], [1], [2], [3]])  # lowest value is 0.8

    # check if alpha is adapted correctly
    model, _ = wsabil_adapt
    model.set_data(X_new, Y_new)

    assert model.adapt_alpha
    assert isclose(model.bound, 0.64)  # 0.8 * min(Y_new)

    # check if alpha stays fixed
    model, _ = wsabil_fixed
    old_alpha = model.bound
    assert not model.adapt_alpha
    assert model.bound == old_alpha


# === tests shared by bounded BQ and WSABI
@pytest.mark.parametrize('bounded_bq', models_test_list + wsabi_test_list)
def test_bounded_bq_shapes(bounded_bq):
    model, _ = bounded_bq

    # integrate
    res = model.integrate()
    assert len(res) == 2
    assert isinstance(res[0], float)
    # None is returned temporarily until the variance is implemented.
    assert res[1] is None

    # transformations
    Y = np.array([[1], [2], [3]])
    assert model.transform(Y).shape == Y.shape
    assert model.inverse_transform(Y).shape == Y.shape

    # predictions base
    x = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 4]])

    res = model.predict_base(x)
    assert len(res) == 4
    for i in range(4):
        assert res[i].shape == (x.shape[0], 1)

    # predictions base full covariance
    res = model.predict_base_with_full_covariance(x)
    assert len(res) == 4
    assert res[0].shape == (x.shape[0], 1)
    assert res[1].shape == (x.shape[0], x.shape[0])
    assert res[2].shape == (x.shape[0], 1)
    assert res[3].shape == (x.shape[0], x.shape[0])

    # predictions
    res = model.predict(x)
    assert len(res) == 2
    assert res[0].shape == (x.shape[0], 1)
    assert res[1].shape == (x.shape[0], 1)

    # predictions full covariance
    res = model.predict_with_full_covariance(x)
    assert len(res) == 2
    assert res[0].shape == (x.shape[0], 1)
    assert res[1].shape == (x.shape[0], x.shape[0])

    # predict gradients
    res = model.get_prediction_gradients(x)
    assert len(res) == 2
    assert res[0].shape == (x.shape[0], x.shape[1])
    assert res[1].shape == (x.shape[0], x.shape[1])


@pytest.mark.parametrize('bounded_bq', models_test_list + wsabi_test_list)
def test_bounded_bq_transformations(bounded_bq):
    model, _ = bounded_bq

    # check if warping and inverse warping correctly yield identity.
    Y = np.array([[1], [2], [3]])
    assert_allclose(model.inverse_transform(model.transform(Y)), Y)
    assert_allclose(model.transform(model.inverse_transform(Y)), Y)

    # check if warping between base GP and model is consistent.
    Y2 = model.Y
    Y1 = model.base_gp.Y
    assert_allclose(model.transform(Y1), Y2)
    assert_allclose(model.inverse_transform(Y2), Y1)


def test_bounded_bq_lower_integrate(bounded_bq_lower):
    # to check the integral, we check if it lies in some confidence interval.
    # these intervals were computed as follows: the mean bounded_bq_lower.predict (first argument) was integrated by
    # simple random sampling with 1e6 samples. Samples were obtained by sampling from the integration measure.
    # The intervals reported are mean\pm 3 std of the 100 integrals obtained by sampling. There might be a very small
    # chance that the true integrals lies outside the specified intervals.
    model, _ = bounded_bq_lower
    interval_mean = [0.6812122058594842, 0.6835927100095945]
    integral_value, integral_variance = model.integrate()
    assert interval_mean[0] < integral_value < interval_mean[1]
    # variance not tested as it is not implemented yet


def test_bounded_bq_upper_integrate(bounded_bq_upper):
    # see test_bounded_bq_lower_integrate function on how the interval was obtained
    model, _ = bounded_bq_upper
    interval_mean = [2.9160404441753704, 2.92034136962288]
    integral_value, integral_variance = model.integrate()
    assert interval_mean[0] < integral_value < interval_mean[1]
    # variance not tested as it is not implemented yet


@pytest.mark.parametrize('bounded_bq', models_test_list + wsabi_test_list)
def test_bounded_bq_gradients(bounded_bq):
    model, _ = bounded_bq
    D = model.X.shape[1]
    N = 4
    x = np.reshape(np.random.randn(D * N), [N, D])

    # mean
    mean_func = lambda z: model.predict(z)[0]
    mean_grad_func = lambda z: model.get_prediction_gradients(z)[0]
    _check_grad(mean_func, mean_grad_func, x)

    # var
    var_func = lambda z: model.predict(z)[1]
    var_grad_func = lambda z: model.get_prediction_gradients(z)[1]
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
