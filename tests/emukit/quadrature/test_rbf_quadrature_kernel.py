# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import GPy
import pytest

from emukit.model_wrappers.gpy_quadrature_wrappers import RBFGPy
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure, QuadratureRBFIsoGaussMeasure
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure


@pytest.fixture
def qrbf_lebesgue_measure():
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
def qrbf_iso_gauss_measure():
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


def test_rbf_qkernel_lebesgue_shapes(qrbf_lebesgue_measure):
    emukit_qrbf, x1, x2, M1, M2, D = qrbf_lebesgue_measure

    # kernel shapes
    assert emukit_qrbf.K(x1, x2).shape == (M1, M2)
    assert emukit_qrbf.qK(x2).shape == (1, M2)
    assert emukit_qrbf.Kq(x1).shape == (M1, 1)
    assert isinstance(emukit_qrbf.qKq(), float)

    # gradient shapes
    assert emukit_qrbf.dKq_dx(x1).shape == (M1, D)
    assert emukit_qrbf.dqK_dx(x2).shape == (D, M2)


def test_rbf_qkernel_lebesgue_qK(qrbf_lebesgue_measure):
    emukit_qrbf, _, x2, _, _, _ = qrbf_lebesgue_measure
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


def test_qkernel_lebesgue_qKq(qrbf_lebesgue_measure):
    emukit_qrbf = qrbf_lebesgue_measure[0]
    # interval was computed as in test_rbf_qkernel_no_measure_qK
    interval = [71.95519933967581, 72.05007241434173]

    qKq = emukit_qrbf.qKq()
    assert interval[0] < qKq < interval[1]


def test_rbf_qkernel_iso_gauss_shapes(qrbf_iso_gauss_measure):
    emukit_qrbf, x1, x2, M1, M2, D = qrbf_iso_gauss_measure

    # kernel shapes
    assert emukit_qrbf.K(x1, x2).shape == (M1, M2)
    assert emukit_qrbf.qK(x2).shape == (1, M2)
    assert emukit_qrbf.Kq(x1).shape == (M1, 1)
    assert isinstance(emukit_qrbf.qKq(), float)

    # gradient shapes
    assert emukit_qrbf.dKq_dx(x1).shape == (M1, D)
    assert emukit_qrbf.dqK_dx(x2).shape == (D, M2)


def test_rbf_qkernel_iso_gauss_qK(qrbf_iso_gauss_measure):
    emukit_qrbf, x1, x2, M1, M2, D = qrbf_iso_gauss_measure
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


def test_qkernel_iso_gauss_qKq(qrbf_iso_gauss_measure):
    emukit_qrbf = qrbf_iso_gauss_measure[0]
    # interval was computed as in test_rbf_qkernel_iso_gauss_qK
    interval = [0.19975038300858916, 0.20025772185633567]

    qKq = emukit_qrbf.qKq()
    assert interval[0] < qKq < interval[1]
