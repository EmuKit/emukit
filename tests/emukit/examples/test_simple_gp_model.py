import numpy as np
import pytest

from emukit.model_wrappers.simple_gp_model import SimpleGaussianProcessModel


@pytest.fixture
def x():
    return np.linspace(-3, 3, 20)[:, None]


@pytest.fixture
def simple_gp(x, y):
    gp = SimpleGaussianProcessModel(x, y)
    return gp


@pytest.fixture
def y(x):
    return x**2


def test_simple_gp_model_predict(simple_gp, x, y):
    simple_gp.optimize()
    mean, var = simple_gp.predict(x)

    assert mean.shape == x.shape
    assert var.shape == x.shape
    # predicting at training locations should give mean results close to training targets
    assert np.allclose(mean, y, atol=1e-3, rtol=1e-3)


def test_set_new_data(simple_gp, x, y):
    simple_gp.optimize()
    mean, var = simple_gp.predict(x)

    simple_gp.set_data(simple_gp.X, 2 * simple_gp.Y)

    mean_2, var_2 = simple_gp.predict(x)

    assert np.allclose(2 * mean, mean_2)
    assert np.allclose(var, var_2)
