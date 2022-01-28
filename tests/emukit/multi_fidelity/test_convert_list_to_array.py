import numpy as np
import pytest

from emukit.multi_fidelity.convert_lists_to_array import (
    convert_x_list_to_array,
    convert_xy_lists_to_arrays,
    convert_y_list_to_array,
)


def test_convert_x_list_to_array():
    x_list = [np.array([[1, 0], [2, 1]]), np.array([[3, 2], [4, 5]])]
    x_array = convert_x_list_to_array(x_list)
    expected_output = np.array([[1, 0, 0], [2, 1, 0], [3, 2, 1], [4, 5, 1]])
    assert np.array_equal(x_array, expected_output)


def test_convert_y_list_to_array():
    y_list = [np.array([[0.0], [1.0]]), np.array([[2.0], [5.0]])]
    y_array = convert_y_list_to_array(y_list)
    expected_output = np.array([[0.0], [1.0], [2.0], [5.0]])
    assert np.array_equal(y_array, expected_output)


def test_convert_xy_lists_to_arrays():
    x_list = [np.array([[1, 0], [2, 1]]), np.array([[3, 2], [4, 5]])]
    y_list = [np.array([[0.0], [1.0]]), np.array([[2.0], [5.0]])]
    x_array, y_array = convert_xy_lists_to_arrays(x_list, y_list)

    expected_y = np.array([[0.0], [1.0], [2.0], [5.0]])
    expected_x = np.array([[1, 0, 0], [2, 1, 0], [3, 2, 1], [4, 5, 1]])
    assert np.array_equal(y_array, expected_y)
    assert np.array_equal(x_array, expected_x)


def test_convert_y_list_to_array_fails_with_1d_input():
    y_list = [np.array([0.0, 1.0]), np.array([2.0, 5.0])]
    with pytest.raises(ValueError):
        convert_y_list_to_array(y_list)


def test_convert_x_list_to_array_fails_with_1d_input():
    x_list = [np.array([0.0, 1.0]), np.array([2.0, 5.0])]
    with pytest.raises(ValueError):
        convert_x_list_to_array(x_list)


def test_convert_xy_lists_to_arrays_fails_with_different_number_of_fidelities():
    x_list = [np.array([[1, 0], [2, 1]]), np.array([[3, 2], [4, 5]])]
    y_list = [np.array([0.0, 1.0]), np.array([2.0, 5.0]), np.array([3, 6])]
    with pytest.raises(ValueError):
        convert_xy_lists_to_arrays(x_list, y_list)


def test_convert_xy_lists_to_arrays_fails_with_different_number_of_points_at_fidelity():
    x_list = [np.array([[1, 0], [2, 1], [3, 4]]), np.array([[3, 2], [4, 5]])]
    y_list = [np.array([0.0, 1.0]), np.array([2.0, 5.0])]
    with pytest.raises(ValueError):
        convert_xy_lists_to_arrays(x_list, y_list)
