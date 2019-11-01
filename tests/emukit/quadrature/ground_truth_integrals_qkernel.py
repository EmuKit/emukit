# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Use this script for ground truth integrals of the quadrature kernels.

import numpy as np
import GPy
from typing import List, Tuple

from emukit.model_wrappers.gpy_quadrature_wrappers import RBFGPy
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure, QuadratureRBFIsoGaussMeasure
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure


# samplers
def _sample_uniform(num_samples: int, bounds: List[Tuple[float, float]]):
    D = len(bounds)
    samples = np.reshape(np.random.rand(num_samples * D), [num_samples, D])
    samples_shifted = np.zeros(samples.shape)
    for d in range(D):
        samples_shifted[:, d] = samples[:, d] * (bounds[d][1] - bounds[d][0]) + bounds[d][0]
    return samples_shifted


def _sample_gauss_iso(num_samples: int, measure: IsotropicGaussianMeasure):
    samples = np.reshape(np.random.randn(num_samples * measure.dim), [num_samples, D])
    return measure.mean + np.sqrt(measure.variance) * samples


# qK integrals
def qK_lebesgue_measure(num_samples: int, qrbf: QuadratureRBFLebesgueMeasure, x2: np.ndarray):
    bounds = qrbf.integral_bounds._bounds
    samples = _sample_uniform(num_samples, bounds)
    Kx = qrbf.K(samples, x2)
    differences = np.array([x[1] - x[0] for x in bounds])
    volume = np.prod(differences)
    return np.mean(Kx, axis=0) * volume


def qK_gauss_iso(num_samples: int, measure: IsotropicGaussianMeasure, qrbf: QuadratureRBFIsoGaussMeasure, x2: np.ndarray):
    samples = _sample_gauss_iso(num_samples, measure)
    Kx = qrbf.K(samples, x2)
    return np.mean(Kx, axis=0)


# qKq integrals
def qKq_lebesgue_measure(num_samples: int, qrbf: QuadratureRBFLebesgueMeasure):
    bounds = qrbf.integral_bounds._bounds
    samples = _sample_uniform(num_samples, bounds)
    qKx = qrbf.qK(samples)
    differences = np.array([x[1] - x[0] for x in bounds])
    volume = np.prod(differences)
    return np.mean(qKx) * volume


def qKq_gauss_iso(num_samples: int, measure: IsotropicGaussianMeasure, qrbf: QuadratureRBFIsoGaussMeasure):
    samples = _sample_gauss_iso(num_samples, measure)
    qKx = qrbf.qK(samples)
    return np.mean(qKx)


if __name__ == "__main__":
    np.random.seed(0)

    # === Choose MEASURE BELOW ======
    MEASURE = 'Lebesgue'
    #MEASURE = 'GaussIso'
    # === CHOOSE MEASURE ABOVE ======

    x1 = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    x2 = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 3]])
    M1 = x1.shape[0]
    M2 = x2.shape[0]
    D = x1.shape[1]

    gpy_kernel = GPy.kern.RBF(input_dim=D)
    emukit_rbf = RBFGPy(gpy_kernel)

    if MEASURE == 'Lebesgue':
        bounds = [(-1, 2), (-3, 3)]  # integral bounds
        emukit_qrbf = QuadratureRBFLebesgueMeasure(emukit_rbf, integral_bounds=bounds)

    elif MEASURE == 'GaussIso':
        measure = IsotropicGaussianMeasure(mean=np.arange(D), variance=2.)
        emukit_qrbf = QuadratureRBFIsoGaussMeasure(rbf_kernel=emukit_rbf, measure=measure)
    else:
        raise ValueError('Measure not defined.')

    print()
    print('measure: {}'.format(MEASURE))
    print('no dimensions: {}'.format(D))
    print()

    # === qK ==============================================================
    num_runs = 100
    num_samples = 1e6
    num_std = 3

    qK_SAMPLES = np.zeros([num_runs, x2.shape[0]])
    qK = emukit_qrbf.qK(x2)[0, :]
    for i in range(num_runs):
        num_samples = int(num_samples)

        if MEASURE == 'Lebesgue':
            qK_samples = qK_lebesgue_measure(num_samples, emukit_qrbf, x2)
        elif MEASURE == 'GaussIso':
            qK_samples = qK_gauss_iso(num_samples, measure, emukit_qrbf, x2)
        else:
            raise ValueError('Measure not defined')

        qK_SAMPLES[i, :] = qK_samples

    print('=== qK ========================================================')
    print('no samples per integral: {:.1E}'.format(num_samples))
    print('number of integrals: {}'.format(num_runs))
    print('number of standard deviations: {}'.format(num_std))
    for i in range(x2.shape[0]):
        print([qK_SAMPLES[:, i].mean() - num_std * qK_SAMPLES[:, i].std(),
               qK_SAMPLES[:, i].mean() + num_std * qK_SAMPLES[:, i].std()])
    print()

    # === qKq =============================================================
    num_runs = 100
    num_samples = 1e6
    num_std = 3

    qKq_SAMPLES = np.zeros(num_runs)
    qKq = emukit_qrbf.qKq()
    for i in range(num_runs):
        num_samples = int(num_samples)

        if MEASURE == 'Lebesgue':
            qKq_samples = qKq_lebesgue_measure(num_samples, emukit_qrbf)
        elif MEASURE == 'GaussIso':
            qKq_samples = qKq_gauss_iso(num_samples, measure, emukit_qrbf)
        else:
            raise ValueError('Measure not defined')

        qKq_SAMPLES[i] = qKq_samples

    print('=== qKq =======================================================')
    print('no samples per integral: {:.1E}'.format(num_samples))
    print('number of integrals: {}'.format(num_runs))
    print('number of standard deviations: {}'.format(num_std))
    print([qKq_SAMPLES.mean() - num_std * qKq_SAMPLES.std(),
           qKq_SAMPLES.mean() + num_std * qKq_SAMPLES.std()])
    print()
