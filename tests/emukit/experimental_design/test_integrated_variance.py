import numpy as np
import pytest

from GPy.models import GPRegression
from GPy.kern import RBF

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.model_wrappers import GPyModelWrapper
from emukit.experimental_design.model_based.acquisitions.integrated_variance import IntegratedVarianceReduction


@pytest.fixture
def model():
    x_init = np.random.rand(5, 2)
    y_init = np.random.rand(5, 1)
    model = GPRegression(x_init, y_init, RBF(2))
    return GPyModelWrapper(model)


def test_integrated_variance_acquisition(model):
    space = ParameterSpace([ContinuousParameter('x1', 0, 1), ContinuousParameter('x2', 0, 1)])

    acquisition = IntegratedVarianceReduction(model, space)

    x_test = np.random.rand(10, 2)
    result = acquisition.evaluate(x_test)
    assert(result.shape == (10, 1))
    assert(np.all(result > 0))


def test_integrated_variance_fails_with_out_of_domain_test_points(model):
    """
    Checks that if the user supplies x_monte_carlo to the function, and they are out of the domain of the parameter space a ValueError is raised.
    """
    space = ParameterSpace([ContinuousParameter('x1', 0, 1), ContinuousParameter('x2', 0, 1)])

    x_monte_carlo = np.array([[0.5, 20.], [0.2, 0.3]])

    with pytest.raises(ValueError):
        IntegratedVarianceReduction(model, space, x_monte_carlo)
