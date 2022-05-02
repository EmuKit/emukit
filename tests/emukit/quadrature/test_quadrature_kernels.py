# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import GPy
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from utils import check_grad, sample_uniform

from emukit.model_wrappers.gpy_quadrature_wrappers import BrownianGPy, ProductMatern32GPy, RBFGPy
from emukit.quadrature.interfaces import IStandardKernel
from emukit.quadrature.kernels import (
    QuadratureBrownianLebesgueMeasure,
    QuadratureKernel,
    QuadratureProductMatern32LebesgueMeasure,
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
    ell = 0.8
    var = 0.5
    kern = RBFGPy(GPy.kern.RBF(input_dim=2, lengthscale=ell, variance=var))


@dataclass
class EmukitProductMatern32:
    variance = 0.7
    lengthscales = np.array([0.4, 1.2])
    kern = ProductMatern32GPy(lengthscales=lengthscales)


@dataclass
class EmukitBrownian:
    var = 0.5
    kern = BrownianGPy(GPy.kern.Brownian(input_dim=1, variance=var))


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


def get_qbrownian_lebesque():
    dat = DataLebesqueSDElike()
    qkern = QuadratureBrownianLebesgueMeasure(EmukitBrownian().kern, integral_bounds=dat.integral_bounds)
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
def qbrownian_lebesgue():
    qkern, dat = get_qbrownian_lebesque()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D, dat.dat_bounds


embeddings_test_list = [
    lazy_fixture("qrbf_lebesgue"),
    lazy_fixture("qrbf_gauss_iso"),
    lazy_fixture("qrbf_uniform_infinite"),
    lazy_fixture("qrbf_uniform_finite"),
    lazy_fixture("qmatern32_lebesgue"),
    lazy_fixture("qbrownian_lebesgue"),
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
        (embeddings_test_list[0], [25.44288768595132, 25.477412341193933]),
        (embeddings_test_list[1], [0.06887117418934532, 0.0690632609840969]),
        (embeddings_test_list[2], [0.13248136022581258, 0.13261016559792643]),
        (embeddings_test_list[3], [0.10728920097517262, 0.10840368292018744]),
        (embeddings_test_list[4], [33.6816570527734, 33.726646173769595]),
        (embeddings_test_list[5], [0.6528048146871609, 0.653858667201299]),
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
                    [0.9926741638312642, 1.0051626943298606],
                    [1.7780365301782586, 1.791985583265183],
                    [1.8441910952044895, 1.8580451530301265],
                    [0.9655979101225538, 0.9774348722979349],
                ]
            ),
        ),
        (
            embeddings_test_list[1],
            np.array(
                [
                    [0.09034074640097703, 0.09112080046226054],
                    [0.09528577689218103, 0.09607836957414669],
                    [0.07773935082749345, 0.07851215820581145],
                    [0.06262796599593302, 0.06326617370265221],
                ]
            ),
        ),
        (
            embeddings_test_list[2],
            np.array(
                [
                    [0.001823024771520544, 0.001845887383131982],
                    [0.03276625478518347, 0.03306556124987184],
                    [0.1463640842176958, 0.1472923670864844],
                    [0.09091056723140134, 0.09179787301481163],
                ]
            ),
        ),
        (
            embeddings_test_list[3],
            np.array(
                [
                    [0.03611462576615197, 0.03673576064006068],
                    [0.03274871262181519, 0.03329912802666011],
                    [0.1107611315779423, 0.11240486902543063],
                    [0.07555953336253515, 0.07707036213410368],
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
                    [0.17436285054037512, 0.1743870565968362],
                    [0.3273488543068163, 0.3276377242105884],
                    [0.5394358272402537, 0.5405072045782628],
                    [0.5892821601114948, 0.5906529816223602],
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
    dfunc = lambda x: emukit_qkernel.dKdiag_dx(x1)
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
    wrong_bounds = [(1, 2), (1, 2)]
    with pytest.raises(ValueError):
        QuadratureBrownianLebesgueMeasure(BrownianGPy(GPy.kern.Brownian()), integral_bounds=wrong_bounds)
