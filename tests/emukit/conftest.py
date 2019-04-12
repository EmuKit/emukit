"""
This file is where to put fixtures that are to be shared across different test files
"""

import GPy
import numpy as np
import pytest

from emukit.core import ContinuousParameter, ParameterSpace, OneHotEncoding
from emukit.model_wrappers import GPyModelWrapper


@pytest.fixture
def n_dims():
    # default to 2 dimensional inputs for tests, but can be overridden by individual tests if desired by using the
    # following decorator: @pytest.mark.parametrize('n_dims', [xxx]) where xxx is the number of desired dimensions
    return 2


@pytest.fixture
def gpy_model(n_dims):
    rng = np.random.RandomState(42)
    x_init = rng.rand(5, n_dims)
    y_init = rng.rand(5, 1)
    gpy_model = GPy.models.GPRegression(x_init, y_init, GPy.kern.RBF(n_dims))
    np.random.seed(42)
    gpy_model.randomize()
    return GPyModelWrapper(gpy_model)

@pytest.fixture
def gpy_model_mcmc(n_dims):
    rng = np.random.RandomState(42)
    x_init = rng.rand(5, n_dims)
    y_init = rng.rand(5, 1)
    gpy_model = GPy.models.GPRegression(x_init, y_init, GPy.kern.RBF(n_dims))
    gpy_model.kern.set_prior(GPy.priors.Uniform(0,5))
    np.random.seed(42)
    gpy_model.randomize()
    return GPyModelWrapper(gpy_model)

@pytest.fixture
def continuous_space(n_dims):
    params = [ContinuousParameter('x' + str(i), 0, 1) for i in range(n_dims)]
    return ParameterSpace(params)


@pytest.fixture
def encoding():
    # different types of volcanoes
    return OneHotEncoding(['strato', 'shield', 'dome'])
