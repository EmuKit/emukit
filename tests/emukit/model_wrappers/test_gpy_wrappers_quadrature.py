"""Basic tests for quadrature GPy wrappers."""

import GPy
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from emukit.model_wrappers.gpy_quadrature_wrappers import (
    BaseGaussianProcessGPy,
    BrownianGPy,
    ProductBrownianGPy,
    ProductMatern12GPy,
    ProductMatern32GPy,
    ProductMatern52GPy,
    RBFGPy,
    create_emukit_model_from_gpy_model,
)
from emukit.quadrature.kernels import (
    QuadratureBrownianLebesgueMeasure,
    QuadratureProductBrownianLebesgueMeasure,
    QuadratureProductMatern12LebesgueMeasure,
    QuadratureProductMatern32LebesgueMeasure,
    QuadratureProductMatern52LebesgueMeasure,
    QuadratureRBFGaussianMeasure,
    QuadratureRBFLebesgueMeasure,
)
from emukit.quadrature.measures import GaussianMeasure, LebesgueMeasure


def get_prod_kernel(kernel_type, n_dim):
    k = kernel_type(input_dim=1, active_dims=[0])
    for i in range(1, n_dim):
        k = k * kernel_type(input_dim=1, active_dims=[i])
    return k


def data(n_dim: int):
    return np.ones([3, n_dim]), np.ones([3, 1])


def integral_bounds(n_dim: int):
    return n_dim * [(0, 1)]


def measure_lebesgue(n_dim: int):
    return LebesgueMeasure.from_bounds(bounds=n_dim * [(0, 1)])


def measure_gaussian(n_dim: int):
    return GaussianMeasure(mean=np.ones(n_dim), variance=1.0)


# === dimension fixtures start here
@pytest.fixture
def dim2():
    return 2


@pytest.fixture
def dim1():
    return 1


# === 1D GPy kernel fixtures start here
@pytest.fixture
def gpy_brownian(dim1):
    kernel_type = GPy.kern.Brownian
    return kernel_type(input_dim=dim1), kernel_type, False


@pytest.fixture
def gpy_matern12(dim1):
    kernel_type = GPy.kern.Exponential
    return kernel_type(input_dim=dim1), kernel_type, False


@pytest.fixture
def gpy_matern32(dim1):
    kernel_type = GPy.kern.Matern32
    return kernel_type(input_dim=dim1), kernel_type, False


@pytest.fixture
def gpy_matern52(dim1):
    kernel_type = GPy.kern.Matern52
    return kernel_type(input_dim=dim1), kernel_type, False


# === 2D GPy kernel fixtures start here
@pytest.fixture
def gpy_rbf(dim2):
    kernel_type = GPy.kern.RBF
    return kernel_type(input_dim=dim2), kernel_type, False


@pytest.fixture
def gpy_prodbrownian(dim2):
    kernel_type = GPy.kern.Brownian
    return get_prod_kernel(kernel_type, dim2), kernel_type, True


@pytest.fixture
def gpy_prodmatern12(dim2):
    kernel_type = GPy.kern.Exponential
    return get_prod_kernel(kernel_type, dim2), kernel_type, True


@pytest.fixture
def gpy_prodmatern32(dim2):
    kernel_type = GPy.kern.Matern32
    return get_prod_kernel(kernel_type, dim2), kernel_type, True


@pytest.fixture
def gpy_prodmatern52(dim2):
    kernel_type = GPy.kern.Matern52
    return get_prod_kernel(kernel_type, dim2), kernel_type, True


def get_wrapper_dict(n_dim, measure, gpy_kern, gpy_kernel_wrapper_type, emukit_qkernel_type):
    gpy_kernel, gpy_kernel_type, is_prod = gpy_kern
    return {
        "data": data(n_dim),
        "measure": measure(n_dim),
        "gpy_kernel": gpy_kernel,
        "gpy_kernel_type": gpy_kernel_type,
        "is_prod": is_prod,
        "gpy_kernel_wrapper_type": gpy_kernel_wrapper_type,
        "emukit_qkernel_type": emukit_qkernel_type,
    }


# === RBF wrapper test cases
@pytest.fixture
def wrapper_rbf_1(dim2, gpy_rbf):
    return get_wrapper_dict(dim2, measure_lebesgue, gpy_rbf, RBFGPy, QuadratureRBFLebesgueMeasure)


@pytest.fixture
def wrapper_rbf_2(dim2, gpy_rbf):
    return get_wrapper_dict(dim2, measure_gaussian, gpy_rbf, RBFGPy, QuadratureRBFGaussianMeasure)


# === (product) Brownian wrapper test cases
@pytest.fixture
def wrapper_brownian_1(dim1, gpy_brownian):
    return get_wrapper_dict(dim1, measure_lebesgue, gpy_brownian, BrownianGPy, QuadratureBrownianLebesgueMeasure)


@pytest.fixture
def wrapper_brownian_2(dim2, gpy_prodbrownian):
    return get_wrapper_dict(
        dim2, measure_lebesgue, gpy_prodbrownian, ProductBrownianGPy, QuadratureProductBrownianLebesgueMeasure
    )


# === Product Matern12 wrapper test cases
@pytest.fixture
def wrapper_matern12_1(dim2, gpy_prodmatern12):
    return get_wrapper_dict(
        dim2, measure_lebesgue, gpy_prodmatern12, ProductMatern12GPy, QuadratureProductMatern12LebesgueMeasure
    )


@pytest.fixture
def wrapper_matern12_2(dim1, gpy_matern12):
    return get_wrapper_dict(
        dim1, measure_lebesgue, gpy_matern12, ProductMatern12GPy, QuadratureProductMatern12LebesgueMeasure
    )


# === Product Matern32 wrapper test cases
@pytest.fixture
def wrapper_matern32_1(dim2, gpy_prodmatern32):
    return get_wrapper_dict(
        dim2, measure_lebesgue, gpy_prodmatern32, ProductMatern32GPy, QuadratureProductMatern32LebesgueMeasure
    )


@pytest.fixture
def wrapper_matern32_2(dim1, gpy_matern32):
    return get_wrapper_dict(
        dim1, measure_lebesgue, gpy_matern32, ProductMatern32GPy, QuadratureProductMatern32LebesgueMeasure
    )


# === Product Matern52 wrapper test cases
@pytest.fixture
def wrapper_matern52_1(dim2, gpy_prodmatern52):
    return get_wrapper_dict(
        dim2, measure_lebesgue, gpy_prodmatern52, ProductMatern52GPy, QuadratureProductMatern52LebesgueMeasure
    )


@pytest.fixture
def wrapper_matern52_2(dim1, gpy_matern52):
    return get_wrapper_dict(
        dim1, measure_lebesgue, gpy_matern52, ProductMatern52GPy, QuadratureProductMatern52LebesgueMeasure
    )


gpy_test_list = [
    lazy_fixture("wrapper_rbf_1"),
    lazy_fixture("wrapper_rbf_2"),
    lazy_fixture("wrapper_brownian_1"),
    lazy_fixture("wrapper_brownian_2"),
    lazy_fixture("wrapper_matern12_1"),
    lazy_fixture("wrapper_matern12_2"),
    lazy_fixture("wrapper_matern32_1"),
    lazy_fixture("wrapper_matern32_2"),
    lazy_fixture("wrapper_matern52_1"),
    lazy_fixture("wrapper_matern52_2"),
]


@pytest.mark.parametrize("wrapper", gpy_test_list)
def test_create_emukit_model_from_gpy_model_types(wrapper):
    gpy_model = GPy.models.GPRegression(kernel=wrapper["gpy_kernel"], X=wrapper["data"][0], Y=wrapper["data"][1])
    emukit_gp = create_emukit_model_from_gpy_model(gpy_model=gpy_model, measure=wrapper["measure"])

    assert isinstance(emukit_gp.kern, wrapper["emukit_qkernel_type"])
    assert isinstance(emukit_gp.kern.kern, wrapper["gpy_kernel_wrapper_type"])

    # product kernel
    if wrapper["is_prod"]:
        assert isinstance(wrapper["gpy_kernel"], GPy.kern.Prod)
        for k in wrapper["gpy_kernel"].parameters:
            assert isinstance(k, wrapper["gpy_kernel_type"])
            assert k.input_dim == 1
    else:
        assert isinstance(emukit_gp.gpy_model.kern, wrapper["gpy_kernel_type"])


def test_create_emukit_model_from_gpy_model_raises_warns():
    input_dim = 2
    gpy_kernel = GPy.kern.RBF(input_dim=input_dim)
    gpy_model = GPy.models.GPRegression(kernel=gpy_kernel, X=np.ones([3, input_dim]), Y=np.ones([3, 1]))

    bounds = input_dim * [(0, 1)]
    measure = LebesgueMeasure.from_bounds(bounds=bounds)

    # Neither measure nor bounds given
    with pytest.raises(ValueError):
        create_emukit_model_from_gpy_model(gpy_model=gpy_model)

    # both measure and bounds are given. Bounds will be ignored.
    with pytest.warns(UserWarning):
        create_emukit_model_from_gpy_model(gpy_model=gpy_model, integral_bounds=bounds, measure=measure)


def test_base_gp_gpy_raises(gpy_prodbrownian):
    incompatible_offset = -3

    n_dim = 2
    dat = data(n_dim=n_dim)
    kern = ProductBrownianGPy(variance=1.0, input_dim=n_dim, offset=incompatible_offset)
    measure = LebesgueMeasure.from_bounds(bounds=n_dim * [(0, 1)])
    qkern = QuadratureProductBrownianLebesgueMeasure(brownian_kernel=kern, measure=measure)

    # this GPy model and hence the emukit base_gp wrapper are not compatible with the kernel wrapper
    # for offsets other than zero.
    gpy_model = GPy.models.GPRegression(kernel=kern.gpy_brownian, X=dat[0], Y=dat[1])

    with pytest.raises(ValueError):
        BaseGaussianProcessGPy(kern=qkern, gpy_model=gpy_model)
