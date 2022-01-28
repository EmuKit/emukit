import mock
import numpy as np
import pytest

from emukit.core.acquisition import Acquisition, IntegratedHyperParameterAcquisition
from emukit.core.interfaces import IPriorHyperparameters


class DummyAcquisition(Acquisition):
    def __init__(self):
        pass

    def evaluate(self, x):
        return np.ones(x.shape[0])

    @property
    def has_gradients(self):
        return False


class DummyAcquisitionWithGradients(Acquisition):
    def __init__(self):
        pass

    def evaluate(self, x):
        return np.ones(x.shape[0])

    def evaluate_with_gradients(self, x):
        return np.ones(x.shape[0]), -np.ones(x.shape[0])

    @property
    def has_gradients(self):
        return True


def test_acquisition_adding():
    acquisition_sum = DummyAcquisition() + DummyAcquisition()
    acquisition_value = acquisition_sum.evaluate(np.array([[0]]))
    assert np.array_equal(acquisition_value, np.array([2.0]))


def test_acquisition_adding_with_gradients():
    acquisition_sum = DummyAcquisitionWithGradients() + DummyAcquisitionWithGradients()
    acquisition_value, acquisition_grads = acquisition_sum.evaluate_with_gradients(np.array([[0]]))
    assert np.array_equal(acquisition_value, np.array([2.0]))
    assert np.array_equiv(acquisition_grads, -np.array([2.0]))


def test_acquistion_adding_with_gradients_fails():
    """
    Checks that trying to evaluate gradients when one of the acquisitions doesn't implement them fails
    """
    acquisition_sum = DummyAcquisitionWithGradients() + DummyAcquisition()
    with pytest.raises(NotImplementedError):
        acquisition_sum.evaluate_with_gradients(np.array([0]))


def test_acquisition_multiplying():
    acquisition_sum = DummyAcquisition() * DummyAcquisition()
    acquisition_value = acquisition_sum.evaluate(np.array([[0]]))
    assert np.array_equal(acquisition_value, np.array([1.0]))


def test_acquisition_multiplying_with_gradients():
    acquisition_sum = DummyAcquisitionWithGradients() * DummyAcquisitionWithGradients()
    acquisition_value, acquisition_grads = acquisition_sum.evaluate_with_gradients(np.array([[0]]))
    assert np.array_equal(acquisition_grads, -np.array([2.0]))


def test_acquisition_division():
    acquisition_sum = DummyAcquisition() / DummyAcquisition()
    acquisition_value = acquisition_sum.evaluate(np.array([[0]]))
    assert np.array_equal(acquisition_value, np.array([1.0]))


def test_acquisition_division_with_gradients():
    acquisition_sum = DummyAcquisitionWithGradients() / DummyAcquisitionWithGradients()
    acquisition_value, acquisition_grads = acquisition_sum.evaluate_with_gradients(np.array([[0]]))
    assert np.array_equal(acquisition_grads, np.array([0.0]))


def test_integrated_acquisition_gradients():
    """
    Check that the integrated hyper parameter acquisition "has_gradients" flag reflects the base acquisition function
    :return:
    """
    mock_model = mock.create_autospec(IPriorHyperparameters)
    mock_acquisition = mock.create_autospec(Acquisition)

    # Check if false
    mock_acquisition.has_gradients = False
    mock_acquisition_generator = lambda x: mock_acquisition
    acq = IntegratedHyperParameterAcquisition(mock_model, mock_acquisition_generator)
    assert acq.has_gradients == False

    # Check if true
    mock_acquisition.has_gradients = True
    acq = IntegratedHyperParameterAcquisition(mock_model, mock_acquisition_generator)
    assert acq.has_gradients == True
