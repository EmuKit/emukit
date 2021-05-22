import numpy as np
import pytest
from emukit.bayesian_optimization.acquisitions.expected_improvement import (
    MeanPluginExpectedImprovement,
    ExpectedImprovement,
)
from emukit.core.interfaces import IModel, IModelWithNoise
from unittest.mock import MagicMock
from emukit.model_wrappers import GPyModelWrapper


class MockNoiselessModel(IModel):
    """
    A mock model with zero observation noise (predict() and predict_noiseless() will return the
    same predictive distribution).
    """
    @staticmethod
    def _mean_func(X):
        return np.sin(X * 30 + X ** 2)

    def _var_func(X):
        return np.cos(X * 10) + 1.2
    
    def predict(self, X):
        return self._mean_func(X), self._var_func(X)

    def predict_noiseless(self, X):
        return self.predict(X)


class MockConstantModel(IModel, IModelWithNoise):
    """Model the predicts the same output distribution everywhere"""
    def predict(self, X):
        # Return mean 1 and variance 8
        return np.ones([x.shape[0], 1]), 8 * np.ones([x.shape[0], 1])
    
    def predict_noiseless(self, X):
        # Return mean 1 and variance 1
        return np.ones([x.shape[0], 1]), np.ones([x.shape[0], 1])


def test_mean_plugin_ei_same_as_standard_on_noiseless():
    np.random.seed(42)

    model = MockNoiselessModel()

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
    # The mean at every previously observed point will be 1, hence y_minimum will be 1.0.
    # The predicted values in the batch should all have mean 1 and variance 1
    # The correct expected improvement for Gaussian Y ~ Normal(1, 1), and y_minimum = 1.0 is 0.3989422804014327
    assert pytest.approx(0.3989422804014327, abs=0, rel=1e-7) == acquisition_values
