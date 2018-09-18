import mock
import pytest

import numpy as np
from emukit.core.loop import (OuterLoop, ModelUpdater, StoppingCondition, CandidatePointCalculator, UserFunctionResult,
                              LoopState, UserFunction)

@pytest.fixture
def mock_updater():
    updater = mock.create_autospec(ModelUpdater)
    updater.update.return_value = None

    return updater

@pytest.fixture
def mock_next_point_calculator():
    next_point = np.array([[0]])

    next_point_calculator = mock.create_autospec(CandidatePointCalculator)
    next_point_calculator.compute_next_points.return_value = next_point
    next_point_calculator.return_value.acquisition = mock.PropertyMock(None)

    return next_point_calculator

@pytest.fixture
def mock_user_function():
    user_function_results = [UserFunctionResult(np.array([0]), np.array([0]))]

    user_function = mock.create_autospec(UserFunction)
    user_function.evaluate.return_value = user_function_results

    return user_function

def test_outer_loop(mock_next_point_calculator, mock_updater, mock_user_function):
    """ Example of automatic outer loop """

    stopping_condition = mock.create_autospec(StoppingCondition)
    stopping_condition.should_stop.side_effect = [False, False, True]

    loop = OuterLoop(mock_next_point_calculator, mock_updater)
    loop.run_loop(mock_user_function, stopping_condition)

    assert(loop.loop_state.iteration == 2)
    assert(np.array_equal(loop.loop_state.X, np.array([[0], [0]])))

def test_accept_non_wrapped_function(mock_next_point_calculator, mock_updater):
    stopping_condition = mock.create_autospec(StoppingCondition)
    stopping_condition.should_stop.side_effect = [False, False, True]
    user_function = lambda x: np.array([[0]])

    loop = OuterLoop(mock_next_point_calculator, mock_updater)
    loop.run_loop(user_function, stopping_condition)

    assert(loop.loop_state.iteration == 2)
    assert(np.array_equal(loop.loop_state.X, np.array([[0], [0]])))

def test_default_condition(mock_next_point_calculator, mock_updater, mock_user_function):
    n_iter = 10

    loop = OuterLoop(mock_next_point_calculator, mock_updater)
    loop.run_loop(mock_user_function, n_iter)

    assert(loop.loop_state.iteration == n_iter)

def test_condition_invalid_type(mock_next_point_calculator, mock_updater, mock_user_function):
    invalid_n_iter = 10.0
    loop = OuterLoop(mock_next_point_calculator, mock_updater)

    with pytest.raises(ValueError):
        loop.run_loop(mock_user_function, invalid_n_iter)
