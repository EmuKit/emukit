# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import mock
import numpy as np

from emukit.benchmarking.loop_benchmarking.metrics import (
    CumulativeCostMetric,
    MeanSquaredErrorMetric,
    MinimumObservedValueMetric,
    TimeMetric,
)
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
    mock_loop.model_updaters = [model_updater_mock]

    loop_state = LoopState([])
    loop_state.metrics = dict()

    mse = MeanSquaredErrorMetric(x_test, y_test)
    metric_value = mse.evaluate(mock_loop, loop_state)

    assert metric_value.shape == (2,)


def test_minimum_observed_value_metric():
    x_observations = np.random.rand(50, 2)
    y_observations = np.random.rand(50, 2)

    mock_model = mock.create_autospec(IModel)

    model_updater_mock = mock.create_autospec(ModelUpdater)
    model_updater_mock.model = mock_model
    mock_loop = mock.create_autospec(OuterLoop)
    mock_loop.model_updaters = [model_updater_mock]

    loop_state = create_loop_state(x_observations, y_observations)
    loop_state.metrics = dict()

    metric = MinimumObservedValueMetric()
    metric_value = metric.evaluate(mock_loop, loop_state)

    assert metric_value.shape == (2,)


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

    name = "time"
    metric = TimeMetric(name)
    metric.reset()
    metric_value = metric.evaluate(mock_loop, loop_state)

    assert metric_value.shape == (1,)


def test_cumulative_costs():
    x_observations = np.random.rand(50, 2)
    y_observations = np.random.rand(50, 2)
    c_observations = np.random.rand(50, 1)
    mock_model = mock.create_autospec(IModel)

    model_updater_mock = mock.create_autospec(ModelUpdater)
    model_updater_mock.model = mock_model
    mock_loop = mock.create_autospec(OuterLoop)
    mock_loop.model_updater = model_updater_mock

    loop_state = create_loop_state(x_observations, y_observations, cost=c_observations)
    loop_state.metrics = dict()

    name = "cost"
    metric = CumulativeCostMetric(name)
    metric.reset()
    metric_value = metric.evaluate(mock_loop, loop_state)

    assert metric_value == np.cumsum(c_observations)[-1]
    assert metric_value.shape == (1,)
