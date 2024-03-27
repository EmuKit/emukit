# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from numpy.testing import assert_array_equal

from emukit.core.loop.loop_state import create_loop_state
from emukit.core.loop.user_function_result import UserFunctionResult


def test_create_loop_state():
    x_init = np.array([[1], [2], [3]])
    y_init = np.array([[4], [5], [6]])

    loop_state = create_loop_state(x_init, y_init)

    assert_array_equal(loop_state.X, x_init)
    assert_array_equal(loop_state.Y, y_init)
    assert loop_state.iteration == 0


def test_create_loop_state_with_cost():
    x_init = np.array([[1], [2], [3]])
    y_init = np.array([[4], [5], [6]])
    cost = np.array([[5], [2], [0]])

    loop_state = create_loop_state(x_init, y_init, cost=cost)

    assert_array_equal(loop_state.X, x_init)
    assert_array_equal(loop_state.Y, y_init)
    assert_array_equal(loop_state.cost, cost)
    assert loop_state.iteration == 0


def test_create_loop_error():
    x_init = np.array([[1], [2], [3]])
    y_init = np.array([[4], [5]])

    with pytest.raises(ValueError):
        create_loop_state(x_init, y_init)


def test_loop_state_update():
    x = np.array([[1], [2], [3], [4]])
    y = np.array([[4], [5], [6], [7]])

    loop_state = create_loop_state(x[:3, :], y[:3, :])
    step_result = UserFunctionResult(x[3, :], y[3, :])
    loop_state.update([step_result])

    assert_array_equal(loop_state.X, x)
    assert_array_equal(loop_state.Y, y)
    assert loop_state.iteration == 1


def test_loop_state_update_error():
    x = np.array([[1], [2], [3], [4]])
    y = np.array([[4], [5], [6], [7]])

    loop_state = create_loop_state(x[:3, :], y[:3, :])
    with pytest.raises(ValueError):
        loop_state.update(None)

    with pytest.raises(ValueError):
        loop_state.update([])


def test_create_loop_state_wrong_size_input_fails():
    x = np.array([[1], [2], [3], [4]])
    y = np.array([[4], [5], [6], [7]])
    c = np.array([[4], [5], [6], [7], [8]])

    with pytest.raises(ValueError):
        create_loop_state(x, y, cost=c)
