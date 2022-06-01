# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import GPy
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from utils import check_grad, sample_uniform

from emukit.model_wrappers.gpy_quadrature_wrappers import (
    BrownianGPy,
    ProductBrownianGPy,
    ProductMatern32GPy,
    ProductMatern52GPy,
    RBFGPy,
)
from emukit.quadrature.interfaces import IStandardKernel
from emukit.quadrature.kernels import (
    QuadratureBrownianLebesgueMeasure,
    QuadratureKernel,
    QuadratureProductBrownianLebesgueMeasure,
    QuadratureProductMatern32LebesgueMeasure,
    QuadratureProductMatern52LebesgueMeasure,
    QuadratureRBFIsoGaussMeasure,
    QuadratureRBFLebesgueMeasure,
    QuadratureRBFUniformMeasure,
)
from emukit.quadrature.measures import IsotropicGaussianMeasure, UniformMeasure


# the following classes and functions are also used to compute the ground truth integrals with MC
@dataclass
class DataLebesque:
    D = 2
    integral_bounds = [(-1, 2), (-3, 3)]
    # x1 and x2 must lay inside domain
    x1 = np.array([[-1, 1], [0, 0.1], [0.5, -1.5]])
    x2 = np.array([[-1, 1], [0, 0.2], [0.8, -0.1], [1.3, 2.8]])
    N = 3
    M = 4
    dat_bounds = integral_bounds


@dataclass
class DataLebesqueSDElike:
    D = 1
    integral_bounds = [(0.2, 1.6)]
    # x1 and x2 must lay inside domain
    x1 = np.array([[0.3], [0.8], [1.5]])
    x2 = np.array([[0.25], [0.5], [1.0], [1.2]])
    N = 3
    M = 4
    dat_bounds = integral_bounds


@dataclass
class DataPositiveDomain:
    D = 2
    integral_bounds = [(0.1, 2), (0.2, 3)]
    # x1 and x2 must lay inside domain
    x1 = np.array([[0.3, 1], [0.8, 0.5], [1.5, 2.8]])
    x2 = np.array([[0.25, 1.1], [0.5, 0.3], [1.0, 1.3], [1.2, 1.2]])
    N = 3
    M = 4
    dat_bounds = integral_bounds


@dataclass
class DataGaussIso:
    D = 2
    measure_mean = np.array([0.2, 1.3])
    measure_var = 2.0
    x1 = np.array([[-1, 1], [0, 0.1], [0.5, -1.5]])
    x2 = np.array([[-1, 1], [0, 0.2], [0.8, -0.1], [1.3, 2.8]])
    N = 3
    M = 4
    dat_bounds = [(m - 2 * np.sqrt(2), m + 2 * np.sqrt(2)) for m in measure_mean]


@dataclass
class DataUniformFinite:
    D = 2
    integral_bounds = [(-1, 2), (-3, 3)]
    bounds = [(1, 2), (-4, 2)]
    # x1 and x2 must lay inside dat_bounds
    x1 = np.array([[0.1, 1], [0, 0.1], [0.5, -1.5]])
    x2 = np.array([[0.1, 1], [0, 0.1], [0.8, -0.1], [1.3, 2]])
    N = 3
    M = 4
    dat_bounds = [(-1, 2), (-2, 3)]


@dataclass
class DataUniformInfinite:
    D = 2
    integral_bounds = None
    bounds = [(1, 2), (-4, 2)]
    # x1 and x2 must lay inside domain
    x1 = np.array([[-1, 1], [0, 0.1], [1.5, -1.5]])
    x2 = np.array([[-1, 1], [0, 0.2], [1.8, -0.1], [1.3, 1.8]])
    N = 3
    M = 4
    dat_bounds = bounds


@dataclass
class EmukitRBF:
    ell = np.array([0.8, 1.3])
    var = 0.5
    kern = RBFGPy(GPy.kern.RBF(input_dim=2, lengthscale=ell, variance=var, ARD=True))


@dataclass
class EmukitProductMatern32:
    variance = 0.7
    lengthscales = np.array([0.4, 1.2])
    kern = ProductMatern32GPy(lengthscales=lengthscales)


@dataclass
class EmukitProductMatern52:
    variance = 0.7
    lengthscales = np.array([0.4, 1.2])
    kern = ProductMatern52GPy(lengthscales=lengthscales)


@dataclass
class EmukitBrownian:
    var = 0.5
    kern = BrownianGPy(GPy.kern.Brownian(input_dim=1, variance=var))


@dataclass
class EmukitProductBrownian:
    variance = 0.7
    offset = -1.8
    kern = ProductBrownianGPy(variance=variance, input_dim=2, offset=offset)


def get_qrbf_lebesque():
    dat = DataLebesque()
    qkern = QuadratureRBFLebesgueMeasure(EmukitRBF().kern, integral_bounds=dat.integral_bounds)
    return qkern, dat


def get_qrbf_gauss_iso():
    dat = DataGaussIso()
    measure = IsotropicGaussianMeasure(mean=dat.measure_mean, variance=dat.measure_var)
    qkern = QuadratureRBFIsoGaussMeasure(EmukitRBF().kern, measure=measure)
    return qkern, dat


def get_qrbf_uniform_finite():
    dat = DataUniformFinite()
    measure = UniformMeasure(bounds=dat.bounds)
    qkern = QuadratureRBFUniformMeasure(rbf_kernel=EmukitRBF.kern, integral_bounds=dat.integral_bounds, measure=measure)
    return qkern, dat


def get_qrbf_uniform_infinite():
    dat = DataUniformInfinite()
    measure = UniformMeasure(bounds=dat.bounds)
    qkern = QuadratureRBFUniformMeasure(rbf_kernel=EmukitRBF.kern, integral_bounds=dat.integral_bounds, measure=measure)
    return qkern, dat


def get_qmatern32_lebesque():
    dat = DataLebesque()
    qkern = QuadratureProductMatern32LebesgueMeasure(EmukitProductMatern32().kern, integral_bounds=dat.integral_bounds)
    return qkern, dat


def get_qmatern52_lebesque():
    dat = DataLebesque()
    qkern = QuadratureProductMatern52LebesgueMeasure(EmukitProductMatern52().kern, integral_bounds=dat.integral_bounds)
    return qkern, dat


def get_qbrownian_lebesque():
    dat = DataLebesqueSDElike()
    qkern = QuadratureBrownianLebesgueMeasure(EmukitBrownian().kern, integral_bounds=dat.integral_bounds)
    return qkern, dat


def get_qprodbrownian_lebesque():
    dat = DataPositiveDomain()
    qkern = QuadratureProductBrownianLebesgueMeasure(EmukitProductBrownian().kern, integral_bounds=dat.integral_bounds)
    return qkern, dat


# == fixtures start here
@pytest.fixture
def qrbf_lebesgue():
    qkern, dat = get_qrbf_lebesque()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def qrbf_gauss_iso():
    qkern, dat = get_qrbf_gauss_iso()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def qrbf_uniform_infinite():
    qkern, dat = get_qrbf_uniform_infinite()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def qrbf_uniform_finite():
    qkern, dat = get_qrbf_uniform_finite()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def qmatern32_lebesgue():
    qkern, dat = get_qmatern32_lebesque()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def qmatern52_lebesgue():
    qkern, dat = get_qmatern52_lebesque()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def qbrownian_lebesgue():
    qkern, dat = get_qbrownian_lebesque()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


@pytest.fixture
def qprodbrownian_lebesque():
    qkern, dat = get_qprodbrownian_lebesque()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


embeddings_test_list = [
    lazy_fixture("qrbf_lebesgue"),
    lazy_fixture("qrbf_gauss_iso"),
    lazy_fixture("qrbf_uniform_infinite"),
    lazy_fixture("qrbf_uniform_finite"),
    lazy_fixture("qmatern32_lebesgue"),
    lazy_fixture("qmatern52_lebesgue"),
    lazy_fixture("qbrownian_lebesgue"),
    lazy_fixture("qprodbrownian_lebesque"),
]


@pytest.mark.parametrize("kernel_embedding", embeddings_test_list)
def test_qkernel_shapes(kernel_embedding):
    emukit_qkernel, x1, x2, N, M, D, _ = kernel_embedding

    # kernel shapes
    assert emukit_qkernel.K(x1, x2).shape == (N, M)

    # embedding shapes
    assert emukit_qkernel.qK(x2).shape == (1, M)
    assert emukit_qkernel.Kq(x1).shape == (N, 1)
    assert np.shape(emukit_qkernel.qKq()) == ()
    assert isinstance(emukit_qkernel.qKq(), float)


# === tests for kernel embedding start here


@pytest.mark.parametrize(
    "kernel_embedding,interval",
    [
        (embeddings_test_list[0], [38.267217898004176, 38.32112525041843]),
        (embeddings_test_list[1], [0.10107780172175222, 0.10132860168906656]),
        (embeddings_test_list[2], [0.19925694197982124, 0.19946236996755512]),
        (embeddings_test_list[3], [0.15840594393483423, 0.16003690383217276]),
        (embeddings_test_list[4], [33.6816570527734, 33.726646173769595]),
        (embeddings_test_list[5], [36.311780552275614, 36.36134818184079]),
        (embeddings_test_list[6], [0.6528048146871609, 0.653858667201299]),
        (embeddings_test_list[7], [147.1346099151547, 147.3099172678195]),
    ],
)
def test_qkernel_qKq(kernel_embedding, interval):
    # To test the integral value of the kernel embedding, we check if it lies in some confidence interval.
    # These intervals were computed as follows: The kernel emukit_qkernel.qK was integrated by
    # simple random sampling with 1e6 samples. This was done 100 times. The intervals show mean\pm 3 std of the 100
    # integrals obtained by sampling. There might be a small chance that the true integrals lies outside the
    # specified intervals.
    emukit_qkernel = kernel_embedding[0]
    qKq = emukit_qkernel.qKq()
    assert interval[0] < qKq < interval[1]


@pytest.mark.parametrize(
    "kernel_embedding,intervals",
    [
        (
            embeddings_test_list[0],
            np.array(
                [
                    [1.5229873955135163, 1.537991178335842],
                    [2.828702323003249, 2.843692090253037],
                    [2.937171398518155, 2.9515842742084026],
                    [1.4729457662097771, 1.4865846411248254],
                ]
            ),
        ),
        (
            embeddings_test_list[1],
            np.array(
                [
                    [0.12486974030387826, 0.125750078406417],
                    [0.13992078671537916, 0.14081149568263046],
                    [0.11892561016472149, 0.1197587877306684],
                    [0.09729471848600614, 0.09804968814418509],
                ]
            ),
        ),
        (
            embeddings_test_list[2],
            np.array(
                [
                    [0.0025833432131653527, 0.0026102032690476376],
                    [0.04944675394035977, 0.049787440901278915],
                    [0.2261261172372856, 0.22702516713823856],
                    [0.13864144061870695, 0.13964034282804863],
                ]
            ),
        ),
        (
            embeddings_test_list[3],
            np.array(
                [
                    [0.051114899926392225, 0.051864595281735905],
                    [0.049445661548958596, 0.05012618410666687],
                    [0.16911997207023577, 0.17112895120953275],
                    [0.1230570007633014, 0.12493562466971112],
                ]
            ),
        ),
        (
            embeddings_test_list[4],
            np.array(
                [
                    [1.1759685512006572, 1.1943216573095992],
                    [2.3867473290884305, 2.4085441010798943],
                    [2.414541592697165, 2.435863015779601],
                    [1.3627594837426076, 1.3814476744322477],
                ]
            ),
        ),
        (
            embeddings_test_list[5],
            np.array(
                [
                    [1.26726828845331, 1.2871778987069316],
                    [2.587496144727501, 2.6113016123209984],
                    [2.6135191767118013, 2.636703247174619],
                    [1.4607696933721623, 1.4808984088931223],
                ]
            ),
        ),
        (
            embeddings_test_list[6],
            np.array(
                [
                    [0.17436285054037512, 0.1743870565968362],
                    [0.3273488543068163, 0.3276377242105884],
                    [0.5394358272402537, 0.5405072045782628],
                    [0.5892821601114948, 0.5906529816223602],
                ]
            ),
        ),
        (
            embeddings_test_list[7],
            np.array(
                [
                    [20.968328255845666, 20.979953254056394],
                    [17.64001267572194, 17.645133556673045],
                    [27.768537209548015, 27.794406087839107],
                    [28.16022039838578, 28.189287432609166],
                ]
            ),
        ),
    ],
)
def test_qkernel_qK(kernel_embedding, intervals):
    # See test_qkernel_qKq on how the intervals were computed.
    emukit_qkernel, _, x2, _, _, _, _ = kernel_embedding
    qK = emukit_qkernel.qK(x2)[0, :]
    for i in range(4):
        assert intervals[i, 0] < qK[i] < intervals[i, 1]


def test_qkernel_uniform_finite_correct_box(qrbf_uniform_finite):
    emukit_qkernel = qrbf_uniform_finite[0]
    # integral bounds are [(-1, 2), (-3, 3)]
    # measure bounds are [(1, 2), (-4, 2)]
    # this test checks that the reasonable box is the union of those boxes
    assert emukit_qkernel.reasonable_box.bounds == [(1, 2), (-3, 2)]


# == tests for kernel gradients start here


@pytest.mark.parametrize("kernel_embedding", embeddings_test_list)
def test_qkernel_gradient_shapes(kernel_embedding):
    emukit_qkernel, x1, x2, N, M, D, _ = kernel_embedding

    # gradient of kernel
    assert emukit_qkernel.dK_dx1(x1, x2).shape == (D, N, M)
    assert emukit_qkernel.dK_dx2(x1, x2).shape == (D, N, M)
    assert emukit_qkernel.dKdiag_dx(x1).shape == (D, N)

    # gradient of embeddings
    assert emukit_qkernel.dKq_dx(x1).shape == (N, D)
    assert emukit_qkernel.dqK_dx(x2).shape == (D, M)


@pytest.mark.parametrize("kernel_embedding", embeddings_test_list)
def test_qkernel_gradient_values(kernel_embedding):
    emukit_qkernel, x1, x2, N, M, D, dat_bounds = kernel_embedding

    np.random.seed(42)
    x1 = sample_uniform(in_shape=(N, D), bounds=dat_bounds)
    x2 = sample_uniform(in_shape=(M, D), bounds=dat_bounds)

    # dKdiag_dx
    in_shape = x1.shape
    func = lambda x: np.diag(emukit_qkernel.K(x, x))
    dfunc = lambda x: emukit_qkernel.dKdiag_dx(x)
    check_grad(func, dfunc, in_shape, dat_bounds)

    # dK_dx1
    in_shape = x1.shape
    func = lambda x: emukit_qkernel.K(x, x2)
    dfunc = lambda x: emukit_qkernel.dK_dx1(x, x2)
    check_grad(func, dfunc, in_shape, dat_bounds)

    # dK_dx2
    in_shape = x2.shape
    func = lambda x: emukit_qkernel.K(x1, x)
    dfunc = lambda x: emukit_qkernel.dK_dx2(x1, x)
    check_grad(func, dfunc, in_shape, dat_bounds)

    # dqK_dx
    in_shape = x2.shape
    func = lambda x: emukit_qkernel.qK(x)
    dfunc = lambda x: emukit_qkernel.dqK_dx(x)
    check_grad(func, dfunc, in_shape, dat_bounds)

    # dKq_dx
    in_shape = x1.shape
    func = lambda x: emukit_qkernel.Kq(x).T
    dfunc = lambda x: emukit_qkernel.dKq_dx(x).T
    check_grad(func, dfunc, in_shape, dat_bounds)


# == tests specific to base class start here


def test_qkernel_raises():
    # must provide either measure or bounds
    with pytest.raises(ValueError):
        QuadratureKernel(IStandardKernel(), integral_bounds=None, measure=None)


# == tests specific to Brownian motion kernel starts here


def test_brownian_qkernel_raises():

    # bounds have wrong dimensionality
    wrong_bounds = [(1, 2), (1, 2)]
    with pytest.raises(ValueError):
        QuadratureBrownianLebesgueMeasure(BrownianGPy(GPy.kern.Brownian()), integral_bounds=wrong_bounds)

    # bounds are negative
    wrong_bounds = [(-1, 2)]
    with pytest.raises(ValueError):
        QuadratureBrownianLebesgueMeasure(BrownianGPy(GPy.kern.Brownian()), integral_bounds=wrong_bounds)

    # bounds are smaller thn offset (product kernel)
    offset = -2
    wrong_bounds = [(offset - 1, offset + 1), (offset + 1, offset + 2)]
    with pytest.raises(ValueError):
        QuadratureProductBrownianLebesgueMeasure(
            ProductBrownianGPy(input_dim=2, offset=offset), integral_bounds=wrong_bounds
        )
