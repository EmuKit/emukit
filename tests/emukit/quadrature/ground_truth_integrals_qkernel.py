# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Use this script for ground truth integrals of the quadrature kernels.

from typing import List, Tuple

import numpy as np
from test_quadrature_kernels import (
    get_qrbf_gauss_iso,
    get_qrbf_lebesque,
    get_qrbf_uniform_finite,
    get_qrbf_uniform_infinite,
)

from emukit.quadrature.kernels import QuadratureKernel
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure


def _sample_uniform(num_samples: int, bounds: List[Tuple[float, float]]) -> np.ndarray:
    D = len(bounds)
    samples = np.reshape(np.random.rand(num_samples * D), [num_samples, D])
    samples_shifted = np.zeros(samples.shape)
    for d in range(D):
        samples_shifted[:, d] = samples[:, d] * (bounds[d][1] - bounds[d][0]) + bounds[d][0]
    return samples_shifted


def _sample_gauss_iso(num_samples: int, measure: IsotropicGaussianMeasure) -> np.ndarray:
    D = measure.num_dimensions
    samples = np.reshape(np.random.randn(num_samples * D), [num_samples, D])
    return measure.mean + np.sqrt(measure.variance) * samples


# === MC estimators start here
def qK_lebesgue(num_samples: int, qkern: QuadratureKernel, x2: np.ndarray) -> np.ndarray:
    """MC estimator for kernel mean qK on Lebesgue measure."""
    bounds = qkern.integral_bounds.bounds
    samples = _sample_uniform(num_samples, bounds)
    Kx = qkern.K(samples, x2)
    differences = np.array([x[1] - x[0] for x in bounds])
    volume = np.prod(differences)
    return np.mean(Kx, axis=0) * volume


def qKq_lebesgue(num_samples: int, qkern: QuadratureKernel) -> float:
    """MC estimator for initial error qKq on Lebesgue measure."""
    bounds = qkern.integral_bounds.bounds
    samples = _sample_uniform(num_samples, bounds)
    qKx = qkern.qK(samples)
    differences = np.array([x[1] - x[0] for x in bounds])
    volume = np.prod(differences)
    return np.mean(qKx) * volume


def qK_gauss_iso(num_samples: int, qkern: QuadratureKernel, x2: np.ndarray) -> np.ndarray:
    """MC estimator for kernel mean qK on isotropic Gaussian measure."""
    measure = qkern.measure
    samples = _sample_gauss_iso(num_samples, measure)
    Kx = qkern.K(samples, x2)
    return np.mean(Kx, axis=0)


def qKq_gauss_iso(num_samples: int, qkern: QuadratureKernel) -> float:
    """MC estimator for initial error qKq on isotropic Gaussian measure."""
    measure = qkern.measure
    samples = _sample_gauss_iso(num_samples, measure)
    qKx = qkern.qK(samples)
    return np.mean(qKx)


def qK_uniform(num_samples: int, qkern: QuadratureKernel, x2: np.ndarray) -> np.ndarray:
    """MC estimator for kernel mean qK on uniform measure."""
    if qkern.integral_bounds is None:
        bounds = qkern.measure.bounds
        samples = _sample_uniform(num_samples, bounds)
        Kx = qkern.K(samples, x2)
        return np.mean(Kx, axis=0)
    else:
        bounds = qkern.integral_bounds._bounds
        samples = _sample_uniform(num_samples, bounds)
        Kx = qkern.K(samples, x2) * qkern.measure.compute_density(samples)[:, np.newaxis]
        differences = np.array([x[1] - x[0] for x in bounds])
        volume = np.prod(differences)
        return np.mean(Kx, axis=0) * volume


def qKq_uniform(num_samples: int, qkern: QuadratureKernel) -> float:
    """MC estimator for initial error qKq on uniform measure."""
    if qkern.integral_bounds is None:
        bounds = qkern.measure.bounds
        samples = _sample_uniform(num_samples, bounds)
        qKx = qkern.qK(samples)
        return np.mean(qKx)
    else:
        bounds = qkern.integral_bounds.bounds
        samples = _sample_uniform(num_samples, bounds)
        qKx = qkern.qK(samples) * qkern.measure.compute_density(samples)[np.newaxis, :]
        differences = np.array([x[1] - x[0] for x in bounds])
        volume = np.prod(differences)
        return np.mean(qKx) * volume


if __name__ == "__main__":
    np.random.seed(0)

    # === Choose MEASURE BELOW ======
    # MEASURE_INTBOUNDS = 'Lebesgue-finite'
    # MEASURE_INTBOUNDS = 'GaussIso-infinite'
    # MEASURE_INTBOUNDS = 'Uniform-infinite'
    MEASURE_INTBOUNDS = "Uniform-finite"
    # === CHOOSE MEASURE ABOVE ======

    if MEASURE_INTBOUNDS == "Lebesgue-finite":
        emukit_qkern, dat = get_qrbf_lebesque()
    elif MEASURE_INTBOUNDS == "GaussIso-infinite":
        emukit_qkern, dat = get_qrbf_gauss_iso()
    elif MEASURE_INTBOUNDS == "Uniform-infinite":
        emukit_qkern, dat = get_qrbf_uniform_infinite()
    elif MEASURE_INTBOUNDS == "Uniform-finite":
        emukit_qkern, dat = get_qrbf_uniform_finite()
    else:
        raise ValueError("Measure-integral-bounds combination not defined")

    print()
    print("measure: {}".format(MEASURE_INTBOUNDS))
    print("no dimensions: {}".format(dat.D))
    print()

    # === qK ==============================================================
    num_runs = 100
    num_samples = 1e6
    num_std = 3

    qK_SAMPLES = np.zeros([num_runs, dat.x2.shape[0]])
    qK = emukit_qkern.qK(dat.x2)[0, :]
    for i in range(num_runs):
        num_samples = int(num_samples)

        if MEASURE_INTBOUNDS == "Lebesgue-finite":
            qK_samples = qK_lebesgue(num_samples, emukit_qkern, dat.x2)
        elif MEASURE_INTBOUNDS == "GaussIso-infinite":
            qK_samples = qK_gauss_iso(num_samples, emukit_qkern, dat.x2)
        elif MEASURE_INTBOUNDS == "Uniform-infinite":
            qK_samples = qK_uniform(num_samples, emukit_qkern, dat.x2)
        elif MEASURE_INTBOUNDS == "Uniform-finite":
            qK_samples = qK_uniform(num_samples, emukit_qkern, dat.x2)
        else:
            raise ValueError("Measure-integral-bounds combination not defined")

        qK_SAMPLES[i, :] = qK_samples

    print("=== qK ========================================================")
    print("no samples per integral: {:.1E}".format(num_samples))
    print("number of integrals: {}".format(num_runs))
    print("number of standard deviations: {}".format(num_std))
    for i in range(dat.x2.shape[0]):
        print(
            [
                qK_SAMPLES[:, i].mean() - num_std * qK_SAMPLES[:, i].std(),
                qK_SAMPLES[:, i].mean() + num_std * qK_SAMPLES[:, i].std(),
            ]
        )
    print()

    # === qKq =============================================================
    num_runs = 100
    num_samples = 1e6
    num_std = 3

    qKq_SAMPLES = np.zeros(num_runs)
    qKq = emukit_qkern.qKq()
    for i in range(num_runs):
        num_samples = int(num_samples)

        if MEASURE_INTBOUNDS == "Lebesgue-finite":
            qKq_samples = qKq_lebesgue(num_samples, emukit_qkern)
        elif MEASURE_INTBOUNDS == "GaussIso-infinite":
            qKq_samples = qKq_gauss_iso(num_samples, emukit_qkern)
        elif MEASURE_INTBOUNDS == "Uniform-infinite":
            qKq_samples = qKq_uniform(num_samples, emukit_qkern)
        elif MEASURE_INTBOUNDS == "Uniform-finite":
            qKq_samples = qKq_uniform(num_samples, emukit_qkern)
        else:
            raise ValueError("Measure-integral-bounds combination not defined")

        qKq_SAMPLES[i] = qKq_samples

    print("=== qKq =======================================================")
    print("no samples per integral: {:.1E}".format(num_samples))
    print("number of integrals: {}".format(num_runs))
    print("number of standard deviations: {}".format(num_std))
    print([qKq_SAMPLES.mean() - num_std * qKq_SAMPLES.std(), qKq_SAMPLES.mean() + num_std * qKq_SAMPLES.std()])
    print()
