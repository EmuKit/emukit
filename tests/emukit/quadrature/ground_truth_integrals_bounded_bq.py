# Use this script for ground truth integrals of the Bounded BQ Gaussian process.

from typing import List, Tuple

import GPy
import numpy as np

from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure
from emukit.quadrature.kernels.quadrature_rbf import QuadratureRBFIsoGaussMeasure
from emukit.quadrature.methods import WSABIL, BoundedBayesianQuadrature


def integral_mean_from_measure_samples(num_samples: int, model: BoundedBayesianQuadrature):
    samples = model.measure.get_samples(num_samples=num_samples)
    mean_at_samples, _ = model.predict(samples)
    return np.mean(mean_at_samples)


if __name__ == "__main__":
    np.random.seed(0)

    # choose model here:
    # METHOD = 'Bounded BQ lower'
    # METHOD = 'Bounded BQ upper'
    # METHOD = 'WSABI-l adapt'
    METHOD = 'WSABI-l fixed'

    # the GPy model
    X = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    Y = np.array([[1], [2], [3]])
    D = X.shape[1]
    gpy_model = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=D))

    # the measure
    measure = IsotropicGaussianMeasure(mean=np.array([0.1, 1.8]), variance=0.8)

    # the emukit base GP
    qrbf = QuadratureRBFIsoGaussMeasure(RBFGPy(gpy_model.kern), measure=measure)
    base_gp = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)

    # the emukit bounded BQ model
    if METHOD == 'Bounded BQ lower':
        bound = np.min(base_gp.Y) - 0.5
        model = BoundedBayesianQuadrature(base_gp=base_gp, X=X, Y=Y,
                                          bound=bound,
                                          is_lower_bounded=True)
    elif METHOD == 'Bounded BQ upper':
        bound = np.max(base_gp.Y) + 0.5
        model = BoundedBayesianQuadrature(base_gp=base_gp, X=X, Y=Y,
                                          bound=bound,
                                          is_lower_bounded=False)
    elif METHOD == 'WSABI-l adapt':
        model = WSABIL(base_gp=base_gp, X=base_gp.X, Y=base_gp.Y, adapt_alpha=True)

    elif METHOD == 'WSABI-l fixed':
        model = WSABIL(base_gp=base_gp, X=base_gp.X, Y=base_gp.Y, adapt_alpha=False)

    else:
        raise ValueError

    print()
    print('method: {}'.format(METHOD))
    print('no dimensions: {}'.format(D))
    print()

    # === mean =============================================================
    num_runs = 100
    num_samples = 1e6
    num_std = 3

    mZ_SAMPLES = np.zeros(num_runs)

    mZ, _ = model.integrate()
    for i in range(num_runs):
        num_samples = int(num_samples)

        mZ_samples = integral_mean_from_measure_samples(num_samples, model)
        mZ_SAMPLES[i] = mZ_samples

    print('=== mean =======================================================')
    print('no samples per integral: {:.1E}'.format(num_samples))
    print('number of integrals: {}'.format(num_runs))
    print('number of standard deviations: {}'.format(num_std))
    print('range with given settings: ')
    print([mZ_SAMPLES.mean() - num_std * mZ_SAMPLES.std(),
           mZ_SAMPLES.mean() + num_std * mZ_SAMPLES.std()])
    print()

    # === variance ==========================================================
    # The variance of the integral is not implemented yet.
