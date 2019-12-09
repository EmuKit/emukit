import numpy as np
import mock

from emukit.core.loop import (ConvergenceStoppingCondition,
                              FixedIterationsStoppingCondition,
                              LoopState,
                              StoppingCondition)


class DummyStoppingCondition(StoppingCondition):
    def should_stop(self, loop_state: LoopState) -> bool:
        pass


def test_fixed_iteration_stopping_condition():
    stopping_condition = FixedIterationsStoppingCondition(5)
    loop_state_mock = mock.create_autospec(LoopState)
    loop_state_mock.iteration = 0

    assert(stopping_condition.should_stop(loop_state_mock) is False)

    loop_state_mock = mock.create_autospec(LoopState)
    loop_state_mock.iteration = 5

    assert(stopping_condition.should_stop(loop_state_mock) is True)


def test_convergence_stopping_condition():
    stopping_condition = ConvergenceStoppingCondition(0.1)

    # check if we stop before criterion can be calculated
    loop_state_mock = mock.create_autospec(LoopState)
    loop_state_mock.iteration = 1
    loop_state_mock.X = np.array([[0]])
    assert(stopping_condition.should_stop(loop_state_mock) is False)

    # check if we stop when we should not
    loop_state_mock = mock.create_autospec(LoopState)
    loop_state_mock.iteration = 5
    loop_state_mock.X = np.array([[0], [10], [20], [30], [40]])
    assert(stopping_condition.should_stop(loop_state_mock) is False)

    # check if we stop when we should
    loop_state_mock = mock.create_autospec(LoopState)
    loop_state_mock.iteration = 5
    loop_state_mock.X.return_value(np.array([[0], [1], [2], [3], [3.01]]))
    assert(stopping_condition.should_stop(loop_state_mock) is True)


def test_operations_with_conditions():
    left_condition = DummyStoppingCondition()
    right_condition = DummyStoppingCondition()
    mock_loop_state = mock.create_autospec(LoopState)

    or_condition = left_condition | right_condition
    and_condition = left_condition & right_condition

    left_condition.should_stop = mock.MagicMock(return_value=True)
    right_condition.should_stop = mock.MagicMock(return_value=True)
    assert(or_condition.should_stop(mock_loop_state) is True)
    assert(and_condition.should_stop(mock_loop_state) is True)

    left_condition.should_stop = mock.MagicMock(return_value=True)
    right_condition.should_stop = mock.MagicMock(return_value=False)
    assert(or_condition.should_stop(mock_loop_state) is True)
    assert(and_condition.should_stop(mock_loop_state) is False)

    left_condition.should_stop = mock.MagicMock(return_value=False)
    right_condition.should_stop = mock.MagicMock(return_value=True)
    assert(or_condition.should_stop(mock_loop_state) is True)
    assert(and_condition.should_stop(mock_loop_state) is False)

    left_condition.should_stop = mock.MagicMock(return_value=False)
    right_condition.should_stop = mock.MagicMock(return_value=False)
    assert(or_condition.should_stop(mock_loop_state) is False)
    assert(and_condition.should_stop(mock_loop_state) is False)

    complex_combination = (left_condition | right_condition) & left_condition
    left_condition.should_stop = mock.MagicMock(return_value=False)
    right_condition.should_stop = mock.MagicMock(return_value=True)
    assert(complex_combination.should_stop(mock_loop_state) is False)
