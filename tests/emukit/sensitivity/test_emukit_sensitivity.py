import mock
import pytest
import numpy as np

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.sensitivity.monte_carlo import MonteCarloSensitivity, ModelFreeMonteCarloSensitivity
from emukit.core.interfaces import IModel
from emukit.test_functions.sensitivity import Ishigami



@pytest.fixture
def model():
    model = mock.create_autospec(IModel)
    model.predict.return_value = (0.1*np.ones((3, 5)), 0.5*np.ones((3, 5)))

    return model

def test_model_based_montecarlo_sensitivity(model):

    space = ParameterSpace([ContinuousParameter('x1', -np.pi, np.pi),
                            ContinuousParameter('x2', -np.pi, np.pi),
                            ContinuousParameter('x3', -np.pi, np.pi)])

    num_mc = 1000
    np.random.seed(0)
    sensitivity = MonteCarloSensitivity(model, space)
    sensitivity.generate_samples(1)

    main_sample = np.array([[ 3.10732573, -1.25504469, -2.66820221], [-1.32105416, -1.45224686, -2.06642419],[ 2.41144646, -1.98949844,  1.66646315]])
    fixing_sample = np.array([[-2.93034587, -2.62148462,  1.71694805], [-2.70120457,  2.60061313,  3.00826754], [-2.56283615,  2.53817347, -1.00496868]])

    main_effects, total_effects, total_variance = sensitivity.compute_effects(main_sample = main_sample,fixing_sample=fixing_sample,num_monte_carlo_points = num_mc)

    keys = space.parameter_names

    assert(all(k in main_effects for k in keys))
    assert(all(k in total_effects for k in keys))

def test_montecarlo_sensitivity():

    ishigami = Ishigami(a=5, b=0.1)
    space = ParameterSpace([ContinuousParameter('x1', -np.pi, np.pi),
                            ContinuousParameter('x2', -np.pi, np.pi),
                            ContinuousParameter('x3', -np.pi, np.pi)])

    num_mc = 1000
    np.random.seed(0)
    sensitivity_ishigami = ModelFreeMonteCarloSensitivity(ishigami.fidelity1, space)
    sensitivity_ishigami.generate_samples(1)

    main_sample = np.array([[ 3.10732573, -1.25504469, -2.66820221], [-1.32105416, -1.45224686, -2.06642419],[ 2.41144646, -1.98949844,  1.66646315]])
    fixing_sample = np.array([[-2.93034587, -2.62148462,  1.71694805], [-2.70120457,  2.60061313,  3.00826754], [-2.56283615,  2.53817347, -1.00496868]])

    main_effects, total_effects, total_variance = sensitivity_ishigami.compute_effects(main_sample = main_sample,fixing_sample=fixing_sample,num_monte_carlo_points = num_mc)

    assert np.abs(total_variance - 1.8659466083857994) < 1e-2
    assert np.abs(main_effects['x1'] - (-8.418294458426123)) < 1e-2
    assert np.abs(total_effects['x1'] - 3.7830552468305543) < 1e-2

