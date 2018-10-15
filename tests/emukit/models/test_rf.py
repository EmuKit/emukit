import numpy as np
import pytest

from emukit.models.random_forest import RandomForest


@pytest.fixture
def model():
    rng = np.random.RandomState(42)
    x_init = rng.rand(5, 2)
    y_init = rng.rand(5, 1)
    model = RandomForest(x_init, y_init)
    return model


def test_predict_shape(model):
    rng = np.random.RandomState(43)

    x_test = rng.rand(10, 2)
    m, v = model.predict(x_test)

    assert(m.shape == (10, 1))
    assert(v.shape == (10, 1))


def test_update_data(model):
    rng = np.random.RandomState(43)
    x_new = rng.rand(5, 2)
    y_new = rng.rand(5, 1)
    model.update_data(x_new, y_new)

    assert(model.X.shape == x_new.shape)
    assert(model.Y.shape == y_new.shape)
