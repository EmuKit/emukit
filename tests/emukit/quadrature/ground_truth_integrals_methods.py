# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Use this script for ground truth integrals of the vanilla BQ Gaussian process.

import numpy as np
from test_quadrature_models import (
    get_bounded_bq_lower,
    get_bounded_bq_upper,
    get_vanilla_bq_model,
    get_wsabil_adapt,
    get_wsabil_fixed,
)

from emukit.quadrature.methods import WarpedBayesianQuadratureModel


def mc_integral_mean_from_measure(num_samples: int, model: WarpedBayesianQuadratureModel) -> float:
    """Computes the MC estimator for the integral mean of the model."""
    samples = model.measure.get_samples(num_samples=num_samples)
    gp_mean_at_samples, _ = model.predict(samples)
    return np.mean(gp_mean_at_samples)


def mc_integral_var_from_measure(num_samples: int, model: WarpedBayesianQuadratureModel) -> float:
    """Computes the MC estimator for the integral variance of the model."""
    samples = model.measure.get_samples(num_samples=num_samples)
    _, gp_cov_at_samples = model.predict_with_full_covariance(samples)
    return np.sum(gp_cov_at_samples) / num_samples**2


if __name__ == "__main__":
    np.random.seed(0)

    # === Choose Model BELOW ======
    METHOD = "Vanilla BQ"
    # METHOD = "Bounded BQ lower"
    # METHOD = "Bounded BQ upper"
    # METHOD = "WSABI-l adapt"
    # METHOD = "WSABI-l fixed"
    # === Choose Model ABOCE ======

    if METHOD == "Vanilla BQ":
        model = get_vanilla_bq_model()
    elif METHOD == "Bounded BQ lower":
        model = get_bounded_bq_lower()
    elif METHOD == "Bounded BQ upper":
        model = get_bounded_bq_upper()
    elif METHOD == "WSABI-l adapt":
        model = get_wsabil_adapt()
    elif METHOD == "WSABI-l fixed":
        model = get_wsabil_fixed()
    else:
        raise ValueError("Method not implemented.")

    print()
    print("method: {}".format(METHOD))
    print("no dimensions: {}".format(model.X.shape[1]))
    print()

    # === mean =============================================================
    print()
    print("=== mean =======================================================")

    num_runs = 100
    num_samples = 1e6
    num_std = 3

    mZ_SAMPLES = np.zeros(num_runs)

    mZ, _ = model.integrate()
    for i in range(num_runs):
        num_samples = int(num_samples)

        mZ_samples = mc_integral_mean_from_measure(num_samples, model)
        mZ_SAMPLES[i] = mZ_samples
        print(".", end="", flush=True)

    print("no samples per integral: {:.1E}".format(num_samples))
    print("number of integrals: {}".format(num_runs))
    print("number of standard deviations: {}".format(num_std))
    print([mZ_SAMPLES.mean() - num_std * mZ_SAMPLES.std(), mZ_SAMPLES.mean() + num_std * mZ_SAMPLES.std()])
    print()

    # === variance ==========================================================
    print()
    print("=== variance ==================================================")

    num_runs = 100
    num_samples = int(5 * 1e3)
    num_std = 3

    # Variance is only implemented for vanilla BQ so far.
    if METHOD == "Vanilla BQ":
        vZ_SAMPLES = np.zeros(num_runs)
        _, vZ = model.integrate()
        for i in range(num_runs):

            vZ_samples = mc_integral_var_from_measure(num_samples, model)
            vZ_SAMPLES[i] = vZ_samples
            print(".", end="", flush=True)

        print("no samples per integral: {:.1E}".format(num_samples))
        print("number of integrals: {}".format(num_runs))
        print("number of standard deviations: {}".format(num_std))
        print([vZ_SAMPLES.mean() - num_std * vZ_SAMPLES.std(), vZ_SAMPLES.mean() + num_std * vZ_SAMPLES.std()])
        print()
