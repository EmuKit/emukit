import numpy as np
import pytest

from emukit.core.acquisition import Acquisition


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
    assert np.array_equal(acquisition_value, np.array([2.]))


def test_acquisition_adding_with_gradients():
    acquisition_sum = DummyAcquisitionWithGradients() + DummyAcquisitionWithGradients()
    acquisition_value, acquisition_grads = acquisition_sum.evaluate_with_gradients(np.array([[0]]))
    assert np.array_equal(acquisition_value, np.array([2.]))
    assert np.array_equiv(acquisition_grads, -np.array([2.]))


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
    assert np.array_equal(acquisition_value, np.array([1.]))


def test_acquisition_multiplying_with_gradients():
    acquisition_sum = DummyAcquisitionWithGradients() * DummyAcquisitionWithGradients()
    acquisition_value, acquisition_grads = acquisition_sum.evaluate_with_gradients(np.array([[0]]))
    assert np.array_equal(acquisition_grads, -np.array([2.]))


def test_acquisition_division():
    acquisition_sum = DummyAcquisition() / DummyAcquisition()
    acquisition_value = acquisition_sum.evaluate(np.array([[0]]))
    assert np.array_equal(acquisition_value, np.array([1.]))


def test_acquisition_division_with_gradients():
    acquisition_sum = DummyAcquisitionWithGradients() / DummyAcquisitionWithGradients()
    acquisition_value, acquisition_grads = acquisition_sum.evaluate_with_gradients(np.array([[0]]))
    assert np.array_equal(acquisition_grads, np.array([0.]))
