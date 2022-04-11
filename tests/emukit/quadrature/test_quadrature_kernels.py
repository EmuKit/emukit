# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from math import isclose

import GPy
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from emukit.model_wrappers.gpy_quadrature_wrappers import ProductMatern32GPy, RBFGPy
from emukit.quadrature.kernels import (
    QuadratureProductMatern32LebesgueMeasure,
    QuadratureRBFIsoGaussMeasure,
    QuadratureRBFLebesgueMeasure,
    QuadratureRBFUniformMeasure,
)
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure, UniformMeasure


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


@dataclass
class DataGaussIso:
    D = 2
    measure_mean = np.array([0.2, 1.3])
    measure_var = 2.0
    x1 = np.array([[-1, 1], [0, 0.1], [0.5, -1.5]])
    x2 = np.array([[-1, 1], [0, 0.2], [0.8, -0.1], [1.3, 2.8]])
    N = 3
    M = 4


@dataclass
class DataUniformFinite:
    D = 2
    integral_bounds = [(-1, 2), (-3, 3)]
    bounds = [(1, 2), (-4, 2)]
    # x1 and x2 must lay inside domain
    x1 = np.array([[0.1, 1], [0, 0.1], [0.5, -1.5]])
    x2 = np.array([[0.1, 1], [0, 0.1], [0.8, -0.1], [1.3, 2]])
    N = 3
    M = 4


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


# == fixtures start here
@pytest.fixture
def qrbf_lebesgue():
    qkern, dat = get_qrbf_lebesque()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D


@pytest.fixture
def qrbf_gauss_iso():
    qkern, dat = get_qrbf_gauss_iso()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D


@pytest.fixture
def qrbf_uniform_infinite():
    qkern, dat = get_qrbf_uniform_infinite()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D


@pytest.fixture
def qrbf_uniform_finite():
    qkern, dat = get_qrbf_uniform_finite()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D


@pytest.fixture
def qmatern32_lebesgue():
    qkern, dat = get_qmatern32_lebesque()
    return qkern, dat.x1, dat.x2, dat.N, dat.M, dat.D


embeddings_test_list = [
    lazy_fixture("qrbf_lebesgue"),
    lazy_fixture("qrbf_gauss_iso"),
    lazy_fixture("qrbf_uniform_infinite"),
    lazy_fixture("qrbf_uniform_finite"),
    lazy_fixture("qmatern32_lebesgue"),
]


@pytest.mark.parametrize("kernel_embedding", embeddings_test_list)
def test_qkernel_shapes(kernel_embedding):
    emukit_qkernel, x1, x2, N, M, D = kernel_embedding

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
        (embeddings_test_list[4], [25.44288768595132, 25.477412341193933]),
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
                    [0.9926741638312642, 1.0051626943298606],
                    [1.7780365301782586, 1.791985583265183],
                    [1.8441910952044895, 1.8580451530301265],
                    [0.9655979101225538, 0.9774348722979349],
                ]
            ),
        ),
    ],
)
def test_qkernel_qK(kernel_embedding, intervals):
    # See test_qkernel_qKq on how the intervals were computed.
    emukit_qkernel, _, x2, _, _, _ = kernel_embedding
    qK = emukit_qkernel.qK(x2)[0, :]
    for i in range(4):
        assert intervals[i, 0] < qK[i] < intervals[i, 1]


def test_qkernel_uniform_finite_correct_box(qrbf_uniform_finite):
    emukit_qkernel = qrbf_uniform_finite[0]
    # integral bounds are [(-1, 2), (-3, 3)]
    # measure bounds are [(1, 2), (-4, 2)]
    # this test checks that the reasonable box is the union of those boxes
    assert emukit_qkernel.reasonable_box_bounds.bounds == [(1, 2), (-3, 2)]


# == tests for kernel gradients start here


@pytest.mark.parametrize("kernel_embedding", embeddings_test_list)
def test_qkernel_gradient_shapes(kernel_embedding):
    emukit_qkernel, x1, x2, N, M, D = kernel_embedding

    # gradient of kernel
    assert emukit_qkernel.dK_dx1(x1, x2).shape == (D, N, M)
    assert emukit_qkernel.dK_dx2(x1, x2).shape == (D, N, M)
    assert emukit_qkernel.dKdiag_dx(x1).shape == (D, N)

    # gradient of embeddings
    assert emukit_qkernel.dKq_dx(x1).shape == (N, D)
    assert emukit_qkernel.dqK_dx(x2).shape == (D, M)


@pytest.mark.parametrize("kernel_embedding", embeddings_test_list)
def test_qkernel_gradient_values(kernel_embedding):
    emukit_qkernel, x1, x2, N, M, D = kernel_embedding
    np.random.seed(42)

    x1 = np.random.randn(N, D)
    x2 = np.random.randn(M, D)

    # dKdiag_dx
    in_shape = x1.shape
    func = lambda x: np.diag(emukit_qkernel.K(x, x))
    dfunc = lambda x: emukit_qkernel.dKdiag_dx(x1)
    _check_grad(func, dfunc, in_shape)

    # dK_dx1
    in_shape = x1.shape
    func = lambda x: emukit_qkernel.K(x, x2)
    dfunc = lambda x: emukit_qkernel.dK_dx1(x, x2)
    _check_grad(func, dfunc, in_shape)

    # dK_dx2
    in_shape = x2.shape
    func = lambda x: emukit_qkernel.K(x1, x)
    dfunc = lambda x: emukit_qkernel.dK_dx2(x1, x)
    _check_grad(func, dfunc, in_shape)

    # dqK_dx
    in_shape = x2.shape
    func = lambda x: emukit_qkernel.qK(x)
    dfunc = lambda x: emukit_qkernel.dqK_dx(x)
    _check_grad(func, dfunc, in_shape)

    # dKq_dx
    in_shape = x1.shape
    func = lambda x: emukit_qkernel.Kq(x).T
    dfunc = lambda x: emukit_qkernel.dKq_dx(x).T
    _check_grad(func, dfunc, in_shape)


def _compute_numerical_gradient(func, dfunc, in_shape):
    """Dimension that is being varied must be last dimension."""
    eps = 1e-8
    x = np.random.randn(*in_shape)
    f = func(x)
    df = dfunc(x)
    dft = np.zeros(df.shape)
    for d in range(x.shape[-1]):
        x_tmp = x.copy()
        x_tmp[..., d] = x_tmp[..., d] + eps
        f_tmp = func(x_tmp)
        dft_d = (f_tmp - f) / eps
        dft[d, ...] = dft_d
    return df, dft


def _check_grad(func, dfunc, in_shape):
    """``func`` must return ``np.ndarray`` of shape ``s`` and ``dfunc`` must return
    ``np.ndarray`` of shape ``s + (input_dim, )``."""
    ABS_TOL = 1e-4
    REL_TOL = 1e-5
    df, dft = _compute_numerical_gradient(func, dfunc, in_shape)
    isclose_all = np.array(
        [isclose(grad1, grad2, rel_tol=REL_TOL, abs_tol=ABS_TOL) for grad1, grad2 in zip(df.flatten(), dft.flatten())]
    )
    assert (1 - isclose_all).sum() == 0
