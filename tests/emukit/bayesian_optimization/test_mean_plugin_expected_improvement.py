from unittest.mock import MagicMock

import numpy as np
import pytest

from emukit.bayesian_optimization.acquisitions.expected_improvement import (
    ExpectedImprovement,
    MeanPluginExpectedImprovement,
)
from emukit.core.interfaces import IModel, IModelWithNoise
from emukit.model_wrappers import GPyModelWrapper


class MockIModel(IModel):
    def __init__(self, X, Y):
        self._X = X
        self._Y = Y

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y


def deterministic_test_func(x: np.ndarray) -> np.ndarray:
    return np.sin(x * 30 + x ** 2).sum(axis=-1, keepdims=True)


class MockNoiselessModel(MockIModel, IModelWithNoise):
    """
    A mock model with zero observation noise (predict() and predict_noiseless() will return the
    same predictive distribution).

    This model mocks predictions for the deterministic_test_func() (the mean prediction will
    be the same as function output).
    """

    @staticmethod
    def _mean_func(X):
        return deterministic_test_func(X)

    @staticmethod
    def _var_func(X):
        return (np.cos(X * 10) + 1.2).sum(axis=-1, keepdims=True)

    def predict(self, X):
        return self._mean_func(X), self._var_func(X)

    def predict_noiseless(self, X):
        return self.predict(X)


class MockConstantModel(MockIModel, IModelWithNoise):
    """Model the predicts the same output distribution everywhere"""

    def predict(self, X):
        # Return mean 1 and variance 8
        return np.ones([X.shape[0], 1]), 8 * np.ones([X.shape[0], 1])

    def predict_noiseless(self, X):
        # Return mean 1 and variance 1
        return np.ones([X.shape[0], 1]), np.ones([X.shape[0], 1])


def test_mean_plugin_ei_same_as_standard_on_noiseless():
    np.random.seed(42)
    X = np.random.randn(10, 3)
    Y = deterministic_test_func(X)

    model = MockNoiselessModel(X, Y)

    mean_plugin_ei = MeanPluginExpectedImprovement(model)
    standard_ei = ExpectedImprovement(model)

    x_new = np.random.randn(100, 3)
    ## Assert the two expected improvement are equal
    assert pytest.approx(standard_ei.evaluate(x_new)) == mean_plugin_ei.evaluate(x_new)


def test_mean_plugin_expected_improvement_returns_expected():
    np.random.seed(43)

    X = np.random.randn(10, 3)
    Y = np.random.randn(10, 1)

    model = MockConstantModel(X, Y)

    mean_plugin_ei = MeanPluginExpectedImprovement(model)

    x_new = np.random.randn(100, 3)
    acquisition_values = mean_plugin_ei.evaluate(x_new)
    # The mean at every previously observed point will be 1, hence y_minimum will be 1.0.
    # The predicted values in the batch should all have mean 1 and variance 1
    # The correct expected improvement for Gaussian Y ~ Normal(1, 1), and y_minimum = 1.0 is 0.3989422804014327
    assert pytest.approx(0.3989422804014327, abs=0, rel=1e-7) == acquisition_values
