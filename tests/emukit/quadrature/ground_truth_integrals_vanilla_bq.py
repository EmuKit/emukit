# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Use this script for ground truth integrals of the vanilla BQ Gaussian process.

import numpy as np
import GPy
from typing import List, Tuple

from emukit.model_wrappers.gpy_quadrature_wrappers import RBFGPy, BaseGaussianProcessGPy
from emukit.quadrature.kernels.quadrature_rbf import QuadratureRBFLebesgueMeasure
from emukit.quadrature.methods import VanillaBayesianQuadrature


def _sample_uniform(num_samples: int, bounds: List[Tuple[float, float]]):
    D = len(bounds)
    samples = np.reshape(np.random.rand(num_samples * D), [num_samples, D])
    samples_shifted = np.zeros(samples.shape)
    for d in range(D):
        samples_shifted[:, d] = samples[:, d] * (bounds[d][1] - bounds[d][0]) + bounds[d][0]
    return samples_shifted


def integral_mean_uniform(num_samples: int, model: VanillaBayesianQuadrature):
    bounds = model.integral_bounds.bounds
    samples = _sample_uniform(num_samples, bounds)
    gp_mean_at_samples, _ = model.predict(samples)
    differences = np.array([x[1] - x[0] for x in bounds])
    volume = np.prod(differences)
    return np.mean(gp_mean_at_samples) * volume


def integral_var_uniform(num_samples: int, model: VanillaBayesianQuadrature):
    bounds = model.integral_bounds.bounds
    samples = _sample_uniform(num_samples, bounds)
    _, gp_cov_at_samples = model.predict_with_full_covariance(samples)
    differences = np.array([x[1] - x[0] for x in bounds])
    volume = np.prod(differences)
    return np.sum(gp_cov_at_samples) * (volume / num_samples) ** 2


if __name__ == "__main__":
    np.random.seed(0)

    METHOD = 'Vanilla BQ'

    X = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    Y = np.array([[1], [2], [3]])
    D = X.shape[1]
    integral_bounds = [(-1, 2), (-3, 3)]

    gpy_model = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=D))
    qrbf = QuadratureRBFLebesgueMeasure(RBFGPy(gpy_model.kern), integral_bounds=integral_bounds)
    model = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)

    vanilla_bq = VanillaBayesianQuadrature(base_gp=model, X=X, Y=Y)

    print()
    print('method: {}'.format(METHOD))
    print('no dimensions: {}'.format(D))
    print()

    # === mean =============================================================
    num_runs = 100
    num_samples = 1e6
    num_std = 3

    mZ_SAMPLES = np.zeros(num_runs)

    mZ, _ = vanilla_bq.integrate()
    for i in range(num_runs):
        num_samples = int(num_samples)

        mZ_samples = integral_mean_uniform(num_samples, vanilla_bq)
        mZ_SAMPLES[i] = mZ_samples

    print('=== mean =======================================================')
    print('no samples per integral: {:.1E}'.format(num_samples))
    print('number of integrals: {}'.format(num_runs))
    print('number of standard deviations: {}'.format(num_std))
    print([mZ_SAMPLES.mean() - num_std * mZ_SAMPLES.std(),
           mZ_SAMPLES.mean() + num_std * mZ_SAMPLES.std()])
    print()

    # === variance ==========================================================
    num_runs = 100
    num_samples = int(5 * 1e3)
    num_std = 3

    vZ_SAMPLES = np.zeros(num_runs)
    _, vZ = vanilla_bq.integrate()
    for i in range(num_runs):

        vZ_samples = integral_var_uniform(num_samples, vanilla_bq)
        vZ_SAMPLES[i] = vZ_samples

    print('=== mean =======================================================')
    print('no samples per integral: {:.1E}'.format(num_samples))
    print('number of integrals: {}'.format(num_runs))
    print('number of standard deviations: {}'.format(num_std))
    print([vZ_SAMPLES.mean() - num_std * vZ_SAMPLES.std(),
           vZ_SAMPLES.mean() + num_std * vZ_SAMPLES.std()])
    print()
