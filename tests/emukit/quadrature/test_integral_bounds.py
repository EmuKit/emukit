# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from numpy.testing import assert_array_equal

from emukit.quadrature.measures import BoxDomain


@pytest.fixture
def integral_bounds():
    bounds = 3 * [(-4, 4)]
    integral_bounds = BoxDomain(name="test_name", bounds=bounds)
    return integral_bounds


def test_integral_bounds_values():
    bounds = [(-1, 1), (-2, 0)]
    lower_bounds = np.array([[-1, -2]])
    upper_bounds = np.array([[1, 0]])

    bounds = BoxDomain(name="test_name", bounds=bounds)
    res = bounds._get_lower_and_upper_bounds()
    assert len(res) == 2
    assert np.all(res[0] == lower_bounds)
    assert np.all(res[1] == upper_bounds)

    assert len(bounds.convert_to_list_of_continuous_parameters()) == 2
    assert bounds.name == "test_name"


def test_integral_bounds_wrong_bounds_init():
    bounds_wrong = [(-1, 1), (0, -2)]

    with pytest.raises(ValueError):
        BoxDomain(name="test_name", bounds=bounds_wrong)

    bounds_empty = []
    with pytest.raises(ValueError):
        BoxDomain(name="test_name", bounds=bounds_empty)


def test_integral_bounds_set(integral_bounds):
    new_bounds = 3 * [(-2, 2)]
    new_lower = np.array([[-2, -2, -2]])
    new_upper = np.array([[2, 2, 2]])
    integral_bounds.bounds = new_bounds

    assert_array_equal(integral_bounds.bounds, new_bounds)
    assert_array_equal(integral_bounds.lower_bounds, new_lower)
    assert_array_equal(integral_bounds.upper_bounds, new_upper)


def test_integral_bounds_set_wrong_bounds(integral_bounds):
    # wrong dimensionality
    wrong_bounds = 4 * [(-2, 2)]
    with pytest.raises(ValueError):
        integral_bounds.bounds = wrong_bounds

    # empty bounds
    wrong_bounds = []
    with pytest.raises(ValueError):
        integral_bounds.bounds = wrong_bounds

    # wrong bound values
    wrong_bounds = 3 * [(-2, -3)]
    with pytest.raises(ValueError):
        integral_bounds.bounds = wrong_bounds
