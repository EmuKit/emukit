import numpy as np
import pytest
from numpy.testing import assert_array_equal

from emukit.core.loop import UserFunctionResult, UserFunctionWrapper
from emukit.core.loop.user_function import MultiSourceFunctionWrapper


def test_user_function_wrapper_evaluation_single_output():
    function = lambda x: 2 * x
    function_input = np.array([[1], [2], [3]])
    ufw = UserFunctionWrapper(function)

    output = ufw.evaluate(function_input)

    assert len(output) == function_input.shape[0]
    for i, record in enumerate(output):
        assert_array_equal(output[i].X, function_input[i])
        assert_array_equal(output[i].Y, function(function_input[i]))


def test_user_function_wrapper_callable_single_output():
    function = lambda x: 2 * x
    function_input = np.array([[1], [2], [3]])
    ufw = UserFunctionWrapper(function)

    evaluated_output = ufw.evaluate(function_input)
    called_output = ufw(function_input)
    assert len(evaluated_output) == len(called_output)
    assert all(eo == co for (eo, co) in zip(evaluated_output, called_output))


def test_user_function_wrapper_evaluation_with_cost():
    function = lambda x: (2 * x, np.array([[1]] * x.shape[0]))
    function_input = np.array([[1], [2], [3]])
    ufw = UserFunctionWrapper(function, extra_output_names=["cost"])

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


def test_multi_source_function_wrapper_evaluation_single_output():
    functions = [lambda x: 2 * x, lambda x: 4 * x]
    function_input = np.array([[1, 0], [2, 1], [3, 0], [4, 0], [5, 1]])
    source_index = -1
    msfw = MultiSourceFunctionWrapper(functions, source_index)

    output = msfw.evaluate(function_input)

    assert len(output) == function_input.shape[0]
    for i, record in enumerate(output):
        assert_array_equal(output[i].X, function_input[i])
        this_function = functions[function_input[i, source_index]]
        this_function_input = np.delete(function_input[i], source_index)
        assert_array_equal(output[i].Y, this_function(this_function_input))


def test_multi_source_function_wrapper_evaluation_with_cost():
    functions = [lambda x: (2 * x, np.array([[1]] * x.shape[0])), lambda x: (4 * x, np.array([[2]] * x.shape[0]))]
    function_input = np.array([[1, 0], [2, 1], [3, 0], [4, 0], [5, 1]])
    source_index = -1
    msfw = MultiSourceFunctionWrapper(functions, source_index, extra_output_names=["cost"])

    output = msfw.evaluate(function_input)

    assert len(output) == function_input.shape[0]
    for i, record in enumerate(output):
        assert_array_equal(output[i].X, function_input[i])
        this_function = functions[function_input[i, source_index]]
        this_function_input = np.delete(function_input[i], source_index)
        assert_array_equal(output[i].Y, this_function(this_function_input)[0])
        assert_array_equal(output[i].cost, this_function(this_function_input)[1][0])


def test_multi_source_function_wrapper_evaluation_with_multiple_extra_arguments():
    functions = [
        lambda x: (2 * x, np.array([[1]] * x.shape[0]), np.array([[1]] * x.shape[0])),
        lambda x: (4 * x, np.array([[2]] * x.shape[0]), np.array([[1]] * x.shape[0])),
    ]
    function_input = np.array([[1, 0], [2, 1], [3, 0], [4, 0], [5, 1]])
    source_index = -1
    msfw = MultiSourceFunctionWrapper(functions, source_index, extra_output_names=["cost", "constraint"])

    output = msfw.evaluate(function_input)

    assert len(output) == function_input.shape[0]
    for i, record in enumerate(output):
        assert_array_equal(output[i].X, function_input[i])
        this_function = functions[function_input[i, source_index]]
        this_function_input = np.delete(function_input[i], source_index)
        assert_array_equal(output[i].Y, this_function(this_function_input)[0])
        assert_array_equal(output[i].cost, this_function(this_function_input)[1][0])
        assert_array_equal(output[i].constraint, this_function(this_function_input)[2][0])


def test_multi_source_function_wrapper_invalid_input():
    # invalid input
    with pytest.raises(ValueError):
        functions = [lambda x: 2 * x]
        function_input = np.array([1, 0])
        msfw = MultiSourceFunctionWrapper(functions)
        msfw.evaluate(function_input)

    # invalid function output
    with pytest.raises(ValueError):
        functions = [lambda x: np.array([2])]
        function_input = np.array([[1, 0]])
        msfw = MultiSourceFunctionWrapper(functions)
        msfw.evaluate(function_input)

    # invalid function output type
    with pytest.raises(ValueError):
        functions = [lambda x: [2]]
        function_input = np.array([[1, 0]])
        msfw = MultiSourceFunctionWrapper(functions)
        msfw.evaluate(function_input)


def test_user_function_result_validation():
    # 2d x
    with pytest.raises(ValueError):
        UserFunctionResult(np.array([[1]]), np.array([1]))

    # 2d y
    with pytest.raises(ValueError):
        UserFunctionResult(np.array([1]), np.array([[1]]))

    # 2d cost
    with pytest.raises(ValueError):
        UserFunctionResult(np.array([1]), np.array([1]), cost=np.array([[1]]))


def test_user_function_too_many_outputs_outputs_fails():
    function = lambda x: (2 * x, np.array([1]))
    function_input = np.array([[1], [2], [3]])
    ufw = UserFunctionWrapper(function)
    with pytest.raises(ValueError):
        ufw.evaluate(function_input)


def test_user_function_too_few_outputs_outputs_fails():
    function = lambda x: 2 * x
    function_input = np.array([[1], [2], [3]])
    ufw = UserFunctionWrapper(function, extra_output_names=["cost"])
    with pytest.raises(ValueError):
        ufw.evaluate(function_input)


def test_multi_source_function_wrapper_too_many_outputs_outputs_fails():
    functions = [
        lambda x: (2 * x, np.array([[1]] * x.shape[0]), np.array([[1]] * x.shape[0])),
        lambda x: (4 * x, np.array([[2]] * x.shape[0]), np.array([[1]] * x.shape[0])),
    ]
    function_input = np.array([[1, 0], [2, 1], [3, 0], [4, 0], [5, 1]])
    source_index = -1
    msfw = MultiSourceFunctionWrapper(functions, source_index)

    with pytest.raises(ValueError):
        msfw.evaluate(function_input)


def test_multi_source_function_wrapper_too_few_outputs_outputs_fails():
    functions = [lambda x: 2 * x, lambda x: 4 * x]
    function_input = np.array([[1, 0], [2, 1], [3, 0], [4, 0], [5, 1]])
    source_index = -1
    msfw = MultiSourceFunctionWrapper(functions, source_index, extra_output_names=["cost"])

    with pytest.raises(ValueError):
        msfw.evaluate(function_input)
