# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from numpy.testing import assert_array_equal

from emukit.quadrature.measures import BoxDomain


@pytest.fixture
def box_domain():
    bounds = 3 * [(-4, 4)]
    integral_bounds = BoxDomain(name="test_name", bounds=bounds)
    return integral_bounds


def test_box_domain_values():
    bounds = [(-1, 1), (-2, 0)]
    test_name = "test_name"
    domain = BoxDomain(name=test_name, bounds=bounds)

    # must match to the ones above
    lower_bounds = np.array([-1, -2])
    upper_bounds = np.array([1, 0])

    assert np.all(domain.lower_bounds == lower_bounds)
    assert np.all(domain.upper_bounds == upper_bounds)

    assert len(domain.convert_to_list_of_continuous_parameters()) == 2
    assert domain.name == test_name


def test_box_domain_wrong_bounds():
    bounds_wrong = [(-1, 1), (0, -2)]

    with pytest.raises(ValueError):
        BoxDomain(name="test_name", bounds=bounds_wrong)

    bounds_empty = []
    with pytest.raises(ValueError):
        BoxDomain(name="test_name", bounds=bounds_empty)


def test_box_domain_set_bounds(box_domain):
    new_bounds = 3 * [(-2, 2)]

    # these must match the above bounds
    new_lower = np.array([-2, -2, -2])
    new_upper = np.array([2, 2, 2])

    # set new
    box_domain.bounds = new_bounds

    assert_array_equal(box_domain.bounds, new_bounds)
    assert_array_equal(box_domain.lower_bounds, new_lower)
    assert_array_equal(box_domain.upper_bounds, new_upper)


def test_box_domain_set_bounds_raises(box_domain):
    # wrong dimensionality
    wrong_bounds = 4 * [(-2, 2)]
    with pytest.raises(ValueError):
        box_domain.bounds = wrong_bounds

    # empty bounds
    wrong_bounds = []
    with pytest.raises(ValueError):
        box_domain.bounds = wrong_bounds

    # wrong bound values
    wrong_bounds = 3 * [(-2, -3)]
    with pytest.raises(ValueError):
        box_domain.bounds = wrong_bounds
