import GPy
import numpy as np
import pytest
import scipy.optimize

from emukit.core.acquisition.acquisition_per_cost import CostAcquisition
from emukit.model_wrappers import GPyModelWrapper


@pytest.fixture
def cost_model():
    x = np.random.rand(10, 2)
    y = np.random.rand(10, 1) + 1
    gpy_model = GPy.models.GPRegression(x, y)
    return GPyModelWrapper(gpy_model)


def test_cost_acquisition_shapes(cost_model):
    cost_acquisition = CostAcquisition(cost_model, 1e-6)
    x_eval = np.random.rand(50, 2)
    value = cost_acquisition.evaluate(x_eval)
    assert value.shape == (50, 1)


def test_cost_acquisition_gradient_shapes(cost_model):
    cost_acquisition = CostAcquisition(cost_model, 1e-6)
    x_eval = np.random.rand(50, 2)
    value = cost_acquisition.evaluate_with_gradients(x_eval)
    assert value[0].shape == (50, 1)
    assert value[1].shape == (50, 2)


def test_cost_acquisition_gradients(cost_model):
    cost_acquisition = CostAcquisition(cost_model, -1000)
    x0 = np.zeros(2)
    gradient_norm_error = scipy.optimize.check_grad(
        lambda x: cost_acquisition.evaluate_with_gradients(x[None, :])[0],
        lambda x: cost_acquisition.evaluate_with_gradients(x[None, :])[1], x0)
    assert np.all(gradient_norm_error < 1e-6)


def test_cost_model_gradients(cost_model):
    x0 = np.zeros(2)
    gradient_norm_error = scipy.optimize.check_grad(
        lambda x: cost_model.predict(x[None, :])[0],
        lambda x: cost_model.get_prediction_gradients(x[None, :])[0], x0)
    assert np.all(gradient_norm_error < 1e-6)
