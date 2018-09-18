import mock
import pytest
import numpy as np

from emukit.core.interfaces import IModel, IDifferentiable
from emukit.experimental_design.model_based.acquisitions import ModelVariance


class MockModel(IModel, IDifferentiable):
    pass


@pytest.fixture
def model():
    model = mock.create_autospec(MockModel)
    model.predict.return_value = (0.1*np.ones((1, 1)), 0.5*np.ones((1, 1)))
    model.get_prediction_gradients.return_value = (np.ones((1, 2)), 2*np.ones((1, 2)))

    return model

def test_model_variance(model):
    acquisition = ModelVariance(model)
    acquisition_value = acquisition.evaluate(np.zeros((1, 2)))
    assert(np.isclose(acquisition_value, 0.5))


def test_model_variance_with_gradients(model):
    acquisition = ModelVariance(model)
    acquisition_value, acquisition_gradients = acquisition.evaluate_with_gradients(np.zeros((1, 2)))
    assert(np.isclose(acquisition_value, 0.5))
    assert(np.all(np.isclose(acquisition_gradients, 2*np.ones((1, 2)))))
