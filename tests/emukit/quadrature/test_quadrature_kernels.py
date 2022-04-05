# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import GPy
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from emukit.model_wrappers.gpy_quadrature_wrappers import Matern32GPy, RBFGPy
from emukit.quadrature.kernels import (
    QuadratureRBFIsoGaussMeasure,
    QuadratureRBFLebesgueMeasure,
    QuadratureRBFUniformMeasure,
    QuadratureMatern32LebesgueMeasure,
)
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure, UniformMeasure


@pytest.fixture
def data():
    x1 = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    x2 = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 3]])
    M1, D = x1.shape
    M2 = x2.shape[0]
    return x1, x2, M1, M2, D


# === RBF kernels
@pytest.fixture
def emukit_rbf(data):
    _, _, _, _, D = data
    return RBFGPy(GPy.kern.RBF(input_dim=D))


@pytest.fixture
def qrbf_lebesgue_finite(data, emukit_rbf):
    x1, x2, M1, M2, D = data
    bounds = [(-1, 2), (-3, 3)]
    emukit_qkernel = QuadratureRBFLebesgueMeasure(emukit_rbf, integral_bounds=bounds)
    return emukit_qkernel, x1, x2, M1, M2, D


@pytest.fixture
def qrbf_iso_gauss_infinite(data, emukit_rbf):
    x1, x2, M1, M2, D = data
    measure = IsotropicGaussianMeasure(mean=np.arange(D), variance=2.0)
    emukit_qkernel = QuadratureRBFIsoGaussMeasure(rbf_kernel=emukit_rbf, measure=measure)
    return emukit_qkernel, x1, x2, M1, M2, D


@pytest.fixture
def qrbf_uniform_infinite(data, emukit_rbf):
    x1, x2, M1, M2, D = data

    measure_bounds = [(0, 2), (-4, 3)]
    measure = UniformMeasure(bounds=measure_bounds)
    emukit_qrbf = QuadratureRBFUniformMeasure(rbf_kernel=emukit_rbf, integral_bounds=None, measure=measure)
    return emukit_qrbf, x1, x2, M1, M2, D


@pytest.fixture
def qrbf_uniform_finite(data, emukit_rbf):
    x1, x2, M1, M2, D = data

    measure = UniformMeasure(bounds=[(1, 2), (-4, 2)])
    bounds = [(-1, 2), (-3, 3)]  # integral bounds
    emukit_qrbf = QuadratureRBFUniformMeasure(rbf_kernel=emukit_rbf, integral_bounds=bounds, measure=measure)
    return emukit_qrbf, x1, x2, M1, M2, D


# === Matern32 kernels
@pytest.fixture
def emukit_matern32(data):
    _, _, _, _, D = data
    return Matern32GPy(GPy.kern.Matern32(input_dim=D))


@pytest.fixture
def qmatern32_lebesgue_finite(data, emukit_matern32):
    x1, x2, M1, M2, D = data
    bounds = [(-1, 2), (-3, 3)]
    emukit_qkernel = QuadratureMatern32LebesgueMeasure(emukit_matern32, integral_bounds=bounds)
    return emukit_qkernel, x1, x2, M1, M2, D


embeddings_test_list = [
    lazy_fixture("qrbf_lebesgue_finite"),
    lazy_fixture("qrbf_iso_gauss_infinite"),
    lazy_fixture("qrbf_uniform_infinite"),
    lazy_fixture("qrbf_uniform_finite"),
    lazy_fixture("qmatern32_lebesgue_finite"),
]


@pytest.mark.parametrize("kernel_embedding", embeddings_test_list)
def test_qkernel_shapes(kernel_embedding):
    emukit_qkernel, x1, x2, M1, M2, D = kernel_embedding

    # kernel shapes
    assert emukit_qkernel.K(x1, x2).shape == (M1, M2)
    assert emukit_qkernel.qK(x2).shape == (1, M2)
    assert emukit_qkernel.Kq(x1).shape == (M1, 1)
    assert isinstance(emukit_qkernel.qKq(), float)

    # gradient shapes
    assert emukit_qkernel.dKq_dx(x1).shape == (M1, D)
    assert emukit_qkernel.dqK_dx(x2).shape == (D, M2)


@pytest.mark.parametrize(
    "kernel_embedding,interval",
    [
        (embeddings_test_list[0], [71.95519933967581, 72.05007241434173]),
        (embeddings_test_list[1], [0.19975038300858916, 0.20025772185633567]),
        (embeddings_test_list[2], [0.24224605771012733, 0.24251553613161855]),
        (embeddings_test_list[3], [0.26910154162464783, 0.2718773521646697]),
        (embeddings_test_list[4], [58.75409064868934, 58.82710428355074]),
    ],
)
def test_qkernel_qKq(kernel_embedding, interval):
    # To test the integral, we check if it lies in some confidence interval.
    # These intervals were computed as follows: the kernel emukit_qkernel.K was integrated in the first argument by
    # simple random sampling with 1e6 samples. This was done 100 times. The intervals show mean\pm 3 std of the 100
    # integrals obtained by sampling. There might be a small chance the true integrals lies outside the specified
    # intervals.
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
                    [3.047326312081091, 3.076162828036314],
                    [5.114069393071916, 5.144207478013706],
                    [0.9879234032609836, 1.0000191615928034],
                    [0.07073863074148745, 0.07217298756057355],
                ]
            ),
        ),
        (
            embeddings_test_list[1],
            np.array(
                [
                    [0.28128128187888524, 0.2831094284574598],
                    [0.28135046180349665, 0.28307273575812275],
                    [0.14890780669545667, 0.15015321562978945],
                    [0.037853812661332246, 0.038507854167645676],
                ]
            ),
        ),
        (
            embeddings_test_list[2],
            np.array(
                [
                    [0.06861687415316085, 0.06936924213600677],
                    [0.21308724091498568, 0.21468500857986952],
                    [0.010109845755552724, 0.010244630092969245],
                    [0.00029973020746309673, 0.0003058513296511006],
                ]
            ),
        ),
        (
            embeddings_test_list[3],
            np.array(
                [
                    [0.018699755197549732, 0.019011045460302092],
                    [0.13793221412478165, 0.13983049581168414],
                    [0.0013275949763495015, 0.001350568348018562],
                    [5.124284006599979e-06, 5.30585124276332e-06],
                ]
            ),
        ),
        (
            embeddings_test_list[4],
            np.array(
                [
                    [2.8540840931792766, 2.8777497371940224],
                    [4.5734146983456485, 4.598356434728914],
                    [1.1403303630264245, 1.1500727643635984],
                    [0.17795992869002905, 0.18007616126477957],
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
