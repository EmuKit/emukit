import mock
import numpy as np

from emukit.benchmarking.metrics import MeanSquaredErrorMetric, MinimumObservedValueMetric, TimeMetric
from emukit.core.interfaces import IModel
from emukit.core.loop import LoopState, ModelUpdater, OuterLoop
from emukit.core.loop.loop_state import create_loop_state


def test_mean_squared_error_metric():
    x_test = np.random.rand(50, 2)
    y_test = np.random.rand(50, 2)

    mock_model = mock.create_autospec(IModel)
    mock_model.predict.return_value = (y_test, y_test * 10)
    model_updater_mock = mock.create_autospec(ModelUpdater)
    model_updater_mock.model = mock_model
    mock_loop = mock.create_autospec(OuterLoop)
    mock_loop.model_updater = model_updater_mock

    loop_state = LoopState([])
    loop_state.metrics = dict()

    mse = MeanSquaredErrorMetric(x_test, y_test)
    mse.evaluate(mock_loop, loop_state)

    assert loop_state.metrics['mean_squared_error'][0].shape == (2,)
    assert np.all(np.isclose(loop_state.metrics['mean_squared_error'][0], np.zeros(2)))


def test_minimum_observed_value_metric():
    x_observations = np.random.rand(50, 2)
    y_observations = np.random.rand(50, 2)

    mock_model = mock.create_autospec(IModel)

    model_updater_mock = mock.create_autospec(ModelUpdater)
    model_updater_mock.model = mock_model
    mock_loop = mock.create_autospec(OuterLoop)
    mock_loop.model_updater = model_updater_mock

    loop_state = create_loop_state(x_observations, y_observations)
    loop_state.metrics = dict()

    metric = MinimumObservedValueMetric()
    metric.evaluate(mock_loop, loop_state)

    assert loop_state.metrics['minimum_observed_value'][0].shape == (2,)


def test_time_metric():
    x_observations = np.random.rand(50, 2)
    y_observations = np.random.rand(50, 2)

    mock_model = mock.create_autospec(IModel)

    model_updater_mock = mock.create_autospec(ModelUpdater)
    model_updater_mock.model = mock_model
    mock_loop = mock.create_autospec(OuterLoop)
    mock_loop.model_updater = model_updater_mock

    loop_state = create_loop_state(x_observations, y_observations)
    loop_state.metrics = dict()

    name = 'time'
    metric = TimeMetric(name)
    metric.reset()
    metric.evaluate(mock_loop, loop_state)

    assert len(loop_state.metrics[name]) == 1
