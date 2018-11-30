import GPy
import numpy as np
import pytest

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.model_wrappers import GPyModelWrapper
from emukit.model_wrappers.gpy_quadrature_wrappers import convert_gpy_model_to_emukit_model
from emukit.quadrature.methods import VanillaBayesianQuadrature


@pytest.fixture
def n_dims():
    # default to 2 dimensional inputs for tests, but can be overridden by individual tests if desired
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
def continuous_space(n_dims):
    params = [ContinuousParameter('x' + str(i), 0, 1) for i in range(n_dims)]
    return ParameterSpace(params)


@pytest.fixture
def vanilla_bq_model(gpy_model, continuous_space, n_dims):
    integral_bounds = continuous_space.convert_to_gpyopt_design_space().get_bounds()
    model = convert_gpy_model_to_emukit_model(gpy_model.model, integral_bounds)
    return VanillaBayesianQuadrature(model)
