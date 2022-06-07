# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Use this script for ground truth integrals of the quadrature kernels.

from typing import List, Tuple

import numpy as np
from test_quadrature_kernels import (
    get_gaussian_qrbf,
    get_lebesgue_normalized_qbrownian,
    get_lebesgue_normalized_qmatern32,
    get_lebesgue_normalized_qmatern52,
    get_lebesgue_normalized_qprodbrownian,
    get_lebesgue_normalized_qrbf,
    get_lebesgue_qbrownian,
    get_lebesgue_qmatern32,
    get_lebesgue_qmatern52,
    get_lebesgue_qprodbrownian,
    get_lebesgue_qrbf,
)

from emukit.quadrature.kernels import QuadratureKernel


def _sample_lebesgue(num_samples: int, bounds: List[Tuple[float, float]]) -> np.ndarray:
    D = len(bounds)
    samples = np.reshape(np.random.rand(num_samples * D), [num_samples, D])
    samples_shifted = np.zeros(samples.shape)
    for d in range(D):
        samples_shifted[:, d] = samples[:, d] * (bounds[d][1] - bounds[d][0]) + bounds[d][0]
    return samples_shifted


def _sample_gaussian(num_samples: int, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    D = mean.shape[0]
    samples = np.reshape(np.random.randn(num_samples * D), [num_samples, D])
    return mean + np.sqrt(variance) * samples


# === MC estimators start here
def qK_gaussian(num_samples: int, qkern: QuadratureKernel, x2: np.ndarray) -> np.ndarray:
    """MC estimator for kernel mean qK on Gaussian measure."""
    measure = qkern.measure
    samples = _sample_gaussian(num_samples, mean=measure.mean, variance=measure.variance)
    Kx = qkern.K(samples, x2)
    return np.mean(Kx, axis=0)


def qKq_gaussian(num_samples: int, qkern: QuadratureKernel) -> float:
    """MC estimator for initial error qKq on Gaussian measure."""
    measure = qkern.measure
    samples = _sample_gaussian(num_samples, mean=measure.mean, variance=measure.variance)
    qKx = qkern.qK(samples)
    return np.mean(qKx)


def qK_lebesgue_normalized(num_samples: int, qkern: QuadratureKernel, x2: np.ndarray) -> np.ndarray:
    """MC estimator for kernel mean qK on Lebesgue measure."""
    bounds = qkern.measure.domain.bounds
    samples = _sample_lebesgue(num_samples, bounds=bounds)
    Kx = qkern.K(samples, x2)
    return np.mean(Kx, axis=0)


def qKq_lebesgue_normalized(num_samples: int, qkern: QuadratureKernel) -> float:
    """MC estimator for initial error qKq on Lebesgue measure."""
    bounds = qkern.measure.domain.bounds
    samples = _sample_lebesgue(num_samples, bounds=bounds)
    qKx = qkern.qK(samples)
    return np.mean(qKx)


def qK_lebesgue(num_samples: int, qkern: QuadratureKernel, x2: np.ndarray) -> np.ndarray:
    """MC estimator for kernel mean qK on Lebesgue measure."""
    differences = np.array([x[1] - x[0] for x in qkern.measure.domain.bounds])
    volume = np.prod(differences)
    return qK_lebesgue_normalized(num_samples=num_samples, qkern=qkern, x2=x2) * volume


def qKq_lebesgue(num_samples: int, qkern: QuadratureKernel) -> float:
    """MC estimator for initial error qKq on Lebesgue measure."""
    differences = np.array([x[1] - x[0] for x in qkern.measure.domain.bounds])
    volume = np.prod(differences)
    return qKq_lebesgue_normalized(num_samples=num_samples, qkern=qkern) * volume


if __name__ == "__main__":
    np.random.seed(0)

    # === Choose MEASURE BELOW ======
    # MEASURE = "Gaussian"
    # MEASURE = "Lebesgue"
    MEASURE = "Lebesgue-normalized"
    # === CHOOSE MEASURE ABOVE ======

    # === Choose KERNEL BELOW ======
    # KERNEL = "rbf"
    # KERNEL = "matern32"
    # KERNEL = "matern52"
    # KERNEL = "brownian"
    KERNEL = "prodbrownian"
    # === CHOOSE KERNEL ABOVE ======

    _e = "Kernel embedding not implemented."
    if KERNEL == "rbf":
        if MEASURE == "Gaussian":
            emukit_qkern, dat = get_gaussian_qrbf()
        elif MEASURE == "Lebesgue":
            emukit_qkern, dat = get_lebesgue_qrbf()
        elif MEASURE == "Lebesgue-normalized":
            emukit_qkern, dat = get_lebesgue_normalized_qrbf()
        else:
            raise ValueError(_e)
    elif KERNEL == "matern32":
        if MEASURE == "Lebesgue":
            emukit_qkern, dat = get_lebesgue_qmatern32()
        elif MEASURE == "Lebesgue-normalized":
            emukit_qkern, dat = get_lebesgue_normalized_qmatern32()
        else:
            raise ValueError(_e)
    elif KERNEL == "matern52":
        if MEASURE == "Lebesgue":
            emukit_qkern, dat = get_lebesgue_qmatern52()
        elif MEASURE == "Lebesgue-normalized":
            emukit_qkern, dat = get_lebesgue_normalized_qmatern52()
        else:
            raise ValueError(_e)
    elif KERNEL == "brownian":
        if MEASURE == "Lebesgue":
            emukit_qkern, dat = get_lebesgue_qbrownian()
        elif MEASURE == "Lebesgue-normalized":
            emukit_qkern, dat = get_lebesgue_normalized_qbrownian()
        else:
            raise ValueError(_e)
    elif KERNEL == "prodbrownian":
        if MEASURE == "Lebesgue":
            emukit_qkern, dat = get_lebesgue_qprodbrownian()
        elif MEASURE == "Lebesgue-normalized":
            emukit_qkern, dat = get_lebesgue_normalized_qprodbrownian()
        else:
            raise ValueError(_e)
    else:
        raise ValueError("Kernel unknown.")

    print()
    print("kernel: {}".format(KERNEL))
    print("measure: {}".format(MEASURE))
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

        if MEASURE == "Gaussian":
            qK_samples = qK_gaussian(num_samples, emukit_qkern, dat.x2)
        elif MEASURE == "Lebesgue":
            qK_samples = qK_lebesgue(num_samples, emukit_qkern, dat.x2)
        elif MEASURE == "Lebesgue-normalized":
            qK_samples = qK_lebesgue_normalized(num_samples, emukit_qkern, dat.x2)
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

        if MEASURE == "Gaussian":
            qKq_samples = qKq_gaussian(num_samples, emukit_qkern)
        elif MEASURE == "Lebesgue":
            qKq_samples = qKq_lebesgue(num_samples, emukit_qkern)
        elif MEASURE == "Lebesgue-normalized":
            qKq_samples = qKq_lebesgue_normalized(num_samples, emukit_qkern)
        else:
            raise ValueError("Measure-integral-bounds combination not defined")

        qKq_SAMPLES[i] = qKq_samples

    print("=== qKq =======================================================")
    print("no samples per integral: {:.1E}".format(num_samples))
    print("number of integrals: {}".format(num_runs))
    print("number of standard deviations: {}".format(num_std))
    print([qKq_SAMPLES.mean() - num_std * qKq_SAMPLES.std(), qKq_SAMPLES.mean() + num_std * qKq_SAMPLES.std()])
    print()
