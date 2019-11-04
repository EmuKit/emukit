# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import GPy
import pytest

from emukit.model_wrappers.gpy_quadrature_wrappers import RBFGPy
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure, QuadratureRBFIsoGaussMeasure, QuadratureRBFUniformMeasure
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure, UniformMeasure


@pytest.fixture
def qrbf_lebesgue_finite():
    x1 = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    x2 = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 3]])
    M1 = x1.shape[0]
    M2 = x2.shape[0]
    D = x1.shape[1]

    # integral bounds
    bounds = [(-1, 2), (-3, 3)]

    gpy_kernel = GPy.kern.RBF(input_dim=D)
    emukit_rbf = RBFGPy(gpy_kernel)
    emukit_qrbf = QuadratureRBFLebesgueMeasure(emukit_rbf, integral_bounds=bounds)
    return emukit_qrbf, x1, x2, M1, M2, D


@pytest.fixture
def qrbf_iso_gauss_infinite():
    x1 = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    x2 = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 3]])
    M1 = x1.shape[0]
    M2 = x2.shape[0]
    D = x1.shape[1]

    gpy_kernel = GPy.kern.RBF(input_dim=D)
    emukit_rbf = RBFGPy(gpy_kernel)
    measure = IsotropicGaussianMeasure(mean=np.arange(D), variance=2.)
    emukit_qrbf = QuadratureRBFIsoGaussMeasure(rbf_kernel=emukit_rbf, measure=measure)
    return emukit_qrbf, x1, x2, M1, M2, D


@pytest.fixture
def qrbf_uniform_infinite():
    x1 = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    x2 = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 3]])
    M1 = x1.shape[0]
    M2 = x2.shape[0]
    D = x1.shape[1]

    # measure
    measure_bounds = [(0, 2), (-4, 3)]
    measure = UniformMeasure(bounds=measure_bounds)

    gpy_kernel = GPy.kern.RBF(input_dim=D)
    emukit_rbf = RBFGPy(gpy_kernel)
    emukit_qrbf = QuadratureRBFUniformMeasure(rbf_kernel=emukit_rbf, integral_bounds=None, measure=measure)
    return emukit_qrbf, x1, x2, M1, M2, D


@pytest.fixture
def qrbf_uniform_finite():
    x1 = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    x2 = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 3]])
    M1 = x1.shape[0]
    M2 = x2.shape[0]
    D = x1.shape[1]

    # measure
    measure_bounds = [(1, 2), (-4, 2)]
    measure = UniformMeasure(bounds=measure_bounds)

    # integral bounds
    bounds = [(-1, 2), (-3, 3)]

    gpy_kernel = GPy.kern.RBF(input_dim=D)
    emukit_rbf = RBFGPy(gpy_kernel)
    emukit_qrbf = QuadratureRBFUniformMeasure(rbf_kernel=emukit_rbf, integral_bounds=bounds, measure=measure)
    return emukit_qrbf, x1, x2, M1, M2, D


def test_rbf_qkernel_lebesgue_shapes(qrbf_lebesgue_finite):
    emukit_qrbf, x1, x2, M1, M2, D = qrbf_lebesgue_finite

    # kernel shapes
    assert emukit_qrbf.K(x1, x2).shape == (M1, M2)
    assert emukit_qrbf.qK(x2).shape == (1, M2)
    assert emukit_qrbf.Kq(x1).shape == (M1, 1)
    assert isinstance(emukit_qrbf.qKq(), float)

    # gradient shapes
    assert emukit_qrbf.dKq_dx(x1).shape == (M1, D)
    assert emukit_qrbf.dqK_dx(x2).shape == (D, M2)


def test_rbf_qkernel_lebesgue_qK(qrbf_lebesgue_finite):
    emukit_qrbf, _, x2, _, _, _ = qrbf_lebesgue_finite
    # to check the integral, we check if it lies in some confidence interval.
    # these intervals were computed as follows: the kernel emukit_qrbf.K was integrated in the first argument by
    # simple random sampling with 1e6 samples. This was done 100 times. The intervals show mean\pm 3 std of the 100
    # integrals obtained by sampling. There might be a very small chance the true integrals lies outside the specified
    # intervals.
    intervals = np.array([[3.047326312081091, 3.076162828036314],
                          [5.114069393071916, 5.144207478013706],
                          [0.9879234032609836, 1.0000191615928034],
                          [0.07073863074148745, 0.07217298756057355]])

    qK = emukit_qrbf.qK(x2)[0, :]
    for i in range(4):
        assert intervals[i, 0] < qK[i] < intervals[i, 1]


def test_qkernel_lebesgue_qKq(qrbf_lebesgue_finite):
    emukit_qrbf = qrbf_lebesgue_finite[0]
    # interval was computed as in test_rbf_qkernel_no_measure_qK
    interval = [71.95519933967581, 72.05007241434173]

    qKq = emukit_qrbf.qKq()
    assert interval[0] < qKq < interval[1]


def test_rbf_qkernel_iso_gauss_shapes(qrbf_iso_gauss_infinite):
    emukit_qrbf, x1, x2, M1, M2, D = qrbf_iso_gauss_infinite

    # kernel shapes
    assert emukit_qrbf.K(x1, x2).shape == (M1, M2)
    assert emukit_qrbf.qK(x2).shape == (1, M2)
    assert emukit_qrbf.Kq(x1).shape == (M1, 1)
    assert isinstance(emukit_qrbf.qKq(), float)

    # gradient shapes
    assert emukit_qrbf.dKq_dx(x1).shape == (M1, D)
    assert emukit_qrbf.dqK_dx(x2).shape == (D, M2)


def test_rbf_qkernel_iso_gauss_qK(qrbf_iso_gauss_infinite):
    emukit_qrbf, x1, x2, M1, M2, D = qrbf_iso_gauss_infinite
    # to check the integral, we check if it lies in some confidence interval.
    # these intervals were computed as follows: the kernel emukit_qrbf.K was integrated in the first argument by
    # simple random sampling with 1e6 samples. This was done 100 times. The intervals show mean\pm 3 std of the 100
    # integrals obtained by sampling. There might be a very small chance the true integrals lies outside the specified
    # intervals.
    intervals = np.array([[0.28128128187888524, 0.2831094284574598],
                          [0.28135046180349665, 0.28307273575812275],
                          [0.14890780669545667, 0.15015321562978945],
                          [0.037853812661332246, 0.038507854167645676]])

    qK = emukit_qrbf.qK(x2)[0, :]
    for i in range(4):
        assert intervals[i, 0] < qK[i] < intervals[i, 1]


def test_qkernel_iso_gauss_qKq(qrbf_iso_gauss_infinite):
    emukit_qrbf = qrbf_iso_gauss_infinite[0]
    # interval was computed as in test_rbf_qkernel_iso_gauss_qK
    interval = [0.19975038300858916, 0.20025772185633567]

    qKq = emukit_qrbf.qKq()
    assert interval[0] < qKq < interval[1]


def test_rbf_qkernel_uniform_infinite_shapes(qrbf_uniform_infinite):
    emukit_qrbf, x1, x2, M1, M2, D = qrbf_uniform_infinite

    # kernel shapes
    assert emukit_qrbf.K(x1, x2).shape == (M1, M2)
    assert emukit_qrbf.qK(x2).shape == (1, M2)
    assert emukit_qrbf.Kq(x1).shape == (M1, 1)
    assert isinstance(emukit_qrbf.qKq(), float)

    # gradient shapes
    assert emukit_qrbf.dKq_dx(x1).shape == (M1, D)
    assert emukit_qrbf.dqK_dx(x2).shape == (D, M2)


def test_rbf_qkernel_uniform_infinite_qK(qrbf_uniform_infinite):
    emukit_qrbf, _, x2, _, _, _ = qrbf_uniform_infinite
    # to check the integral, we check if it lies in some confidence interval.
    # these intervals were computed as follows: the kernel emukit_qrbf.K was integrated in the first argument by
    # simple random sampling with 1e6 samples. This was done 100 times. The intervals show mean\pm 3 std of the 100
    # integrals obtained by sampling. There might be a very small chance the true integrals lies outside the specified
    # intervals.
    intervals = np.array([[0.06861687415316085, 0.06936924213600677],
                          [0.21308724091498568, 0.21468500857986952],
                          [0.010109845755552724, 0.010244630092969245],
                          [0.00029973020746309673, 0.0003058513296511006]])

    qK = emukit_qrbf.qK(x2)[0, :]
    for i in range(4):
        assert intervals[i, 0] < qK[i] < intervals[i, 1]


def test_qkernel_uniform_infinite_qKq(qrbf_uniform_infinite):
    emukit_qrbf = qrbf_uniform_infinite[0]
    # interval was computed as in test_rbf_qkernel_no_measure_qK
    interval = [0.24224605771012733, 0.24251553613161855]

    qKq = emukit_qrbf.qKq()
    assert interval[0] < qKq < interval[1]


def test_rbf_qkernel_uniform_finite_shapes(qrbf_uniform_finite):
    emukit_qrbf, x1, x2, M1, M2, D = qrbf_uniform_finite

    # kernel shapes
    assert emukit_qrbf.K(x1, x2).shape == (M1, M2)
    assert emukit_qrbf.qK(x2).shape == (1, M2)
    assert emukit_qrbf.Kq(x1).shape == (M1, 1)
    assert isinstance(emukit_qrbf.qKq(), float)

    # gradient shapes
    assert emukit_qrbf.dKq_dx(x1).shape == (M1, D)
    assert emukit_qrbf.dqK_dx(x2).shape == (D, M2)


def test_rbf_qkernel_uniform_finite_qK(qrbf_uniform_finite):
    emukit_qrbf, _, x2, _, _, _ = qrbf_uniform_finite
    # to check the integral, we check if it lies in some confidence interval.
    # these intervals were computed as follows: the kernel emukit_qrbf.K was integrated in the first argument by
    # simple random sampling with 1e6 samples. This was done 100 times. The intervals show mean\pm 3 std of the 100
    # integrals obtained by sampling. There might be a very small chance the true integrals lies outside the specified
    # intervals.
    intervals = np.array([[0.018699755197549732, 0.019011045460302092],
                          [0.13793221412478165, 0.13983049581168414],
                          [0.0013275949763495015, 0.001350568348018562],
                          [5.124284006599979e-06, 5.30585124276332e-06]])

    qK = emukit_qrbf.qK(x2)[0, :]
    for i in range(4):
        assert intervals[i, 0] < qK[i] < intervals[i, 1]


def test_qkernel_uniform_finite_qKq(qrbf_uniform_finite):
    emukit_qrbf = qrbf_uniform_finite[0]
    # interval was computed as in test_rbf_qkernel_no_measure_qK
    interval = [0.26910154162464783, 0.2718773521646697]

    qKq = emukit_qrbf.qKq()
    assert interval[0] < qKq < interval[1]
