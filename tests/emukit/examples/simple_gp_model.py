import numpy as np

from emukit.examples.simple_gp_model.simple_gp_model import SimpleGaussianProcessModel


def test_simple_gp_model_optimize():
    x = np.linspace(-1, 1, 3)[:, None]
    y = x **2
    gp = SimpleGaussianProcessModel(x, y)
    gp.optimize()


def test_simple_gp_model_predict():
    x = np.linspace(-1, 1, 3)[:, None]
    y = x**2
    gp = SimpleGaussianProcessModel(x, y)
    mean, var = gp.predict(np.array(5, 1))
    assert mean.shape == (5, 1)
    assert var.shape == (5, 1)
