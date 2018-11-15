import mock
import numpy as np
import pytest

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.interfaces import IModel
from emukit.sensitivity.monte_carlo import (ModelFreeMonteCarloSensitivity,
                                            MonteCarloSensitivity)
from emukit.test_functions.sensitivity import Ishigami


@pytest.fixture
def space():
    space = ParameterSpace([ContinuousParameter('x1', 0, 1),
                            ContinuousParameter('x2', 0, 1),
                            ContinuousParameter('x3', 0, 1)])
    return space

def test_model_based_montecarlo_sensitivity(space):

    model = mock.create_autospec(IModel)
    model.predict.return_value = (0.1*np.ones((3, 1)), np.zeros((3, 1)))

    sensitivity = MonteCarloSensitivity(model, space)

    num_mc = 1
    main_sample = np.zeros((3,3))
    fixing_sample = np.zeros((3,3))

    main_effects, total_effects, total_variance = sensitivity.compute_effects(main_sample=main_sample, fixing_sample=fixing_sample, num_monte_carlo_points=num_mc)

    keys = space.parameter_names
    assert(all(k in main_effects for k in keys))
    assert(all(k in total_effects for k in keys))

    eps = 1e-6
    assert (abs(total_variance) < eps), "constant return value should yield 0 variance"


def test_montecarlo_sensitivity(space):

    mock_function = lambda x: np.sin(x[:, 0] + x[:, 1] + x[:, 2])

    sensitivity_ishigami = ModelFreeMonteCarloSensitivity(mock_function, space)

    num_mc = 1
    main_sample = 0.1*np.ones((3,3))
    fixing_sample = np.zeros((3,3))

    main_effects, total_effects, total_variance = sensitivity_ishigami.compute_effects(main_sample=main_sample, fixing_sample=fixing_sample, num_monte_carlo_points=num_mc)

    keys = space.parameter_names
    assert(all(k in main_effects for k in keys))
    assert(all(k in total_effects for k in keys))

    eps = 1e-6
    assert (abs(total_variance) < eps), "constant return value should yield 0 variance"
