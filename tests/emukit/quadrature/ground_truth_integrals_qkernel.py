# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Use this script for ground truth integrals of the quadrature kernels.

import numpy as np
import GPy
from typing import List, Tuple

from emukit.model_wrappers.gpy_quadrature_wrappers import RBFGPy
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure, UniformMeasure
from emukit.quadrature.kernels import QuadratureRBFLebesgueMeasure, QuadratureRBFIsoGaussMeasure, QuadratureRBFUniformMeasure


def _sample_uniform(num_samples: int, bounds: List[Tuple[float, float]]):
    D = len(bounds)
    samples = np.reshape(np.random.rand(num_samples * D), [num_samples, D])
    samples_shifted = np.zeros(samples.shape)
    for d in range(D):
        samples_shifted[:, d] = samples[:, d] * (bounds[d][1] - bounds[d][0]) + bounds[d][0]
    return samples_shifted


def _sample_gauss_iso(num_samples: int, measure: IsotropicGaussianMeasure):
    D = measure.num_dimensions
    samples = np.reshape(np.random.randn(num_samples * D), [num_samples, D])
    return measure.mean + np.sqrt(measure.variance) * samples


def qK_lebesgue(num_samples: int, qrbf: QuadratureRBFLebesgueMeasure, x2: np.ndarray):
    bounds = qrbf.integral_bounds._bounds
    samples = _sample_uniform(num_samples, bounds)
    Kx = qrbf.K(samples, x2)
    differences = np.array([x[1] - x[0] for x in bounds])
    volume = np.prod(differences)
    return np.mean(Kx, axis=0) * volume


def qKq_lebesgue(num_samples: int, qrbf: QuadratureRBFLebesgueMeasure):
    bounds = qrbf.integral_bounds._bounds
    samples = _sample_uniform(num_samples, bounds)
    qKx = qrbf.qK(samples)
    differences = np.array([x[1] - x[0] for x in bounds])
    volume = np.prod(differences)
    return np.mean(qKx) * volume


def qK_gauss_iso(num_samples: int, qrbf: QuadratureRBFIsoGaussMeasure, x2: np.ndarray):
    measure = qrbf.measure
    samples = _sample_gauss_iso(num_samples, measure)
    Kx = qrbf.K(samples, x2)
    return np.mean(Kx, axis=0)


def qKq_gauss_iso(num_samples: int, qrbf: QuadratureRBFIsoGaussMeasure):
    measure = qrbf.measure
    samples = _sample_gauss_iso(num_samples, measure)
    qKx = qrbf.qK(samples)
    return np.mean(qKx)


def qK_uniform(num_samples: int, qrbf: QuadratureRBFUniformMeasure, x2: np.ndarray):
    if qrbf.integral_bounds is None:
        bounds = qrbf.measure.bounds
        samples = _sample_uniform(num_samples, bounds)
        Kx = qrbf.K(samples, x2)
        return np.mean(Kx, axis=0)
    else:
        bounds = qrbf.integral_bounds._bounds
        samples = _sample_uniform(num_samples, bounds)
        Kx = qrbf.K(samples, x2) * qrbf.measure.compute_density(samples)[:, np.newaxis]
        differences = np.array([x[1] - x[0] for x in bounds])
        volume = np.prod(differences)
        return np.mean(Kx, axis=0) * volume


def qKq_uniform(num_samples: int, qrbf: QuadratureRBFUniformMeasure):
    if qrbf.integral_bounds is None:
        bounds = qrbf.measure.bounds
        samples = _sample_uniform(num_samples, bounds)
        qKx = qrbf.qK(samples)
        return np.mean(qKx)
    else:
        bounds = qrbf.integral_bounds._bounds
        samples = _sample_uniform(num_samples, bounds)
        qKx = qrbf.qK(samples) * qrbf.measure.compute_density(samples)[np.newaxis, :]
        differences = np.array([x[1] - x[0] for x in bounds])
        volume = np.prod(differences)
        return np.mean(qKx) * volume


if __name__ == "__main__":
    np.random.seed(0)

    # === Choose MEASURE BELOW ======
    #MEASURE_INTBOUNDS = 'Lebesgue-finite'
    #MEASURE_INTBOUNDS = 'GaussIso-infinite'
    #MEASURE_INTBOUNDS = 'Uniform-infinite'
    MEASURE_INTBOUNDS = 'Uniform-finite'
    # === CHOOSE MEASURE ABOVE ======

    x1 = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    x2 = np.array([[-1, 1], [0, 0], [-2, 0.1], [-3, 3]])
    D = x1.shape[1]

    gpy_kernel = GPy.kern.RBF(input_dim=D)
    emukit_rbf = RBFGPy(gpy_kernel)

    if MEASURE_INTBOUNDS == 'Lebesgue-finite':
        emukit_qrbf = QuadratureRBFLebesgueMeasure(emukit_rbf, integral_bounds=[(-1, 2), (-3, 3)])
    elif MEASURE_INTBOUNDS == 'GaussIso-infinite':
        measure = IsotropicGaussianMeasure(mean=np.arange(D), variance=2.)
        emukit_qrbf = QuadratureRBFIsoGaussMeasure(rbf_kernel=emukit_rbf, measure=measure)
    elif MEASURE_INTBOUNDS == 'Uniform-infinite':
        measure = UniformMeasure(bounds=[(0, 2), (-4, 3)])
        emukit_qrbf = QuadratureRBFUniformMeasure(emukit_rbf, integral_bounds=None, measure=measure)
    elif MEASURE_INTBOUNDS == 'Uniform-finite':
        measure = UniformMeasure(bounds=[(1, 2), (-4, 2)])
        emukit_qrbf = QuadratureRBFUniformMeasure(emukit_rbf, integral_bounds=[(-1, 2), (-3, 3)], measure=measure)
    else:
        raise ValueError('Measure-integral-bounds combination not defined')

    print()
    print('measure: {}'.format(MEASURE_INTBOUNDS))
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

        if MEASURE_INTBOUNDS == 'Lebesgue-finite':
            qK_samples = qK_lebesgue(num_samples, emukit_qrbf, x2)
        elif MEASURE_INTBOUNDS == 'GaussIso-infinite':
            qK_samples = qK_gauss_iso(num_samples, emukit_qrbf, x2)
        elif MEASURE_INTBOUNDS == 'Uniform-infinite':
            qK_samples = qK_uniform(num_samples, emukit_qrbf, x2)
        elif MEASURE_INTBOUNDS == 'Uniform-finite':
            qK_samples = qK_uniform(num_samples, emukit_qrbf, x2)
        else:
            raise ValueError('Measure-integral-bounds combination not defined')

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

        if MEASURE_INTBOUNDS == 'Lebesgue-finite':
            qKq_samples = qKq_lebesgue(num_samples, emukit_qrbf)
        elif MEASURE_INTBOUNDS == 'GaussIso-infinite':
            qKq_samples = qKq_gauss_iso(num_samples, emukit_qrbf)
        elif MEASURE_INTBOUNDS == 'Uniform-infinite':
            qKq_samples = qKq_uniform(num_samples, emukit_qrbf)
        elif MEASURE_INTBOUNDS == 'Uniform-finite':
            qKq_samples = qKq_uniform(num_samples, emukit_qrbf)
        else:
            raise ValueError('Measure-integral-bounds combination not defined')

        qKq_SAMPLES[i] = qKq_samples

    print('=== qKq =======================================================')
    print('no samples per integral: {:.1E}'.format(num_samples))
    print('number of integrals: {}'.format(num_runs))
    print('number of standard deviations: {}'.format(num_std))
    print([qKq_SAMPLES.mean() - num_std * qKq_SAMPLES.std(),
           qKq_SAMPLES.mean() + num_std * qKq_SAMPLES.std()])
    print()
