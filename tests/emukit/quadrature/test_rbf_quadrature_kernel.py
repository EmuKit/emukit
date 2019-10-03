# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import GPy
import pytest

from emukit.model_wrappers.gpy_quadrature_wrappers import RBFGPy
from emukit.quadrature.kernels import QuadratureRBFnoMeasure, QuadratureRBFIsoGaussMeasure
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure


@pytest.fixture
def qrbf_no_measure():
    x1 = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    x2 = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 3]])
    M1 = x1.shape[0]
    M2 = x2.shape[0]
    D = x1.shape[1]

    # integral bounds
    lb = -3
    ub = 3

    gpy_kernel = GPy.kern.RBF(input_dim=D)
    emukit_rbf = RBFGPy(gpy_kernel)
    emukit_qrbf = QuadratureRBFnoMeasure(emukit_rbf, integral_bounds=D * [(lb, ub)])
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
    measure = IsotropicGaussianMeasure(mean=np.zeros(D), variance=1.)
    emukit_qrbf = QuadratureRBFIsoGaussMeasure(rbf_kernel=emukit_rbf, measure=measure)
    return emukit_qrbf, x1, x2, M1, M2, D


def test_rbf_qkernel_no_measure_shapes(qrbf_no_measure):
    emukit_qrbf, x1, x2, M1, M2, D = qrbf_no_measure

    # kernel shapes
    assert emukit_qrbf.K(x1, x2).shape == (M1, M2)
    assert emukit_qrbf.qK(x2).shape == (1, M2)
    assert emukit_qrbf.Kq(x1).shape == (M1, 1)
    assert isinstance(emukit_qrbf.qKq(), float)

    # gradient shapes
    assert emukit_qrbf.dKq_dx(x1).shape == (M1, D)
    assert emukit_qrbf.dqK_dx(x2).shape == (D, M2)


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
