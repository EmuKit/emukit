import pytest
import numpy as np
from numpy.testing import assert_array_equal

from emukit.core.loop import UserFunctionWrapper, UserFunctionResult


def test_user_function_wrapper_evaluation_no_cost():
    function = lambda x: 2 * x
    function_input = np.array([[1], [2], [3]])
    ufw = UserFunctionWrapper(function)

    output = ufw.evaluate(function_input)

    assert len(output) == function_input.shape[0]
    for i, record in enumerate(output):
        assert_array_equal(output[i].X, function_input[i])
        assert_array_equal(output[i].Y, function(function_input[i]))
        assert output[i].cost is None


def test_user_function_wrapper_evaluation_with_cost():
    function = lambda x: (2 * x, np.array([[1]] * x.shape[0]))
    function_input = np.array([[1], [2], [3]])
    ufw = UserFunctionWrapper(function)

    output = ufw.evaluate(function_input)

    assert len(output) == function_input.shape[0]
    for i, record in enumerate(output):
        assert_array_equal(output[i].X, function_input[i])
        assert_array_equal(output[i].Y, function(function_input[i])[0])
        assert_array_equal(output[i].cost, function(function_input[i])[1][0])


def test_user_function_wrapper_invalid_input():
    # invalid input
    with pytest.raises(ValueError):
        function = lambda x: 2 * x
        function_input = np.array([1])
        ufw = UserFunctionWrapper(function)
        ufw.evaluate(function_input)

    # invalid function output
    with pytest.raises(ValueError):
        function = lambda x: np.array([2])
        function_input = np.array([[1]])
        ufw = UserFunctionWrapper(function)
        ufw.evaluate(function_input)

    # invalid function output type
    with pytest.raises(ValueError):
        function = lambda x: [2]
        function_input = np.array([[1]])
        ufw = UserFunctionWrapper(function)
        ufw.evaluate(function_input)


def test_user_function_result_validation():
    # 2d x
    with pytest.raises(ValueError):
        UserFunctionResult(np.array([[1]]), np.array([1]))

    # 2d y
    with pytest.raises(ValueError):
        UserFunctionResult(np.array([1]), np.array([[1]]))

    # 2d cost
    with pytest.raises(ValueError):
        UserFunctionResult(np.array([1]), np.array([1]), np.array([[1]]))
