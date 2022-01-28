import numpy as np
import pytest

try:
    from emukit.examples.models.bohamiann import Bohamiann
except ImportError:
    # Bohamiann has an import issue. See https://github.com/automl/pybnn/pull/1
    pytestmark = pytest.mark.skip


@pytest.fixture
def model():
    rng = np.random.RandomState(42)
    x_init = rng.rand(5, 2)
    y_init = rng.rand(5, 1)
    model = Bohamiann(x_init, y_init, num_burnin=100, num_steps=200)
    return model


def test_predict_shape(model):
    rng = np.random.RandomState(43)

    x_test = rng.rand(10, 2)
    m, v = model.predict(x_test)

    assert m.shape == (10, 1)
    assert v.shape == (10, 1)


def test_update_data(model):
    rng = np.random.RandomState(43)
    x_new = rng.rand(5, 2)
    y_new = rng.rand(5, 1)
    model.set_data(x_new, y_new)

    assert model.X.shape == x_new.shape
    assert model.Y.shape == y_new.shape


def test_get_prediction_gradients_shape(model):
    rng = np.random.RandomState(43)

    x_test = rng.rand(10, 2)
    dm, dv = model.get_prediction_gradients(x_test)

    assert dm.shape == x_test.shape
    assert dv.shape == x_test.shape
