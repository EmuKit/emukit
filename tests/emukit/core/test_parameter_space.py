# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from unittest import mock

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from emukit.core import (
    BanditParameter,
    CategoricalParameter,
    ContinuousParameter,
    DiscreteParameter,
    InformationSourceParameter,
    OneHotEncoding,
    ParameterSpace,
)


@pytest.fixture
def space_2d():
    p1 = ContinuousParameter("c", 1.0, 5.0)
    p2 = ContinuousParameter("d", 1.0, 6.0)

    return ParameterSpace([p1, p2])


@pytest.fixture
def space_3d_mixed():
    p1 = ContinuousParameter("c", 1.0, 5.0)
    p2 = DiscreteParameter("d", [1, 2, 3])
    p3 = CategoricalParameter("cat", OneHotEncoding(["Maine Coon", "Siamese"]))
    return ParameterSpace([p1, p2, p3])


def test_parameter_space_has_all_parameters(space_2d):
    assert len(space_2d.parameters) == 2


def test_check_in_domain(space_2d, space_3d_mixed):
    x_test = np.array([[1.5, 6.1], [1.5, 2.0]])
    in_domain = space_2d.check_points_in_domain(x_test)
    assert np.array_equal(in_domain, np.array([False, True]))

    x_mixed_test = np.array([[1.5, 0, 1, 1], [1.5, 1, 1.0, 0.0]])
    in_domain = space_3d_mixed.check_points_in_domain(x_mixed_test)
    assert np.array_equal(in_domain, np.array([False, True]))


def test_check_in_domain_with_only_bandit_parameters():
    p1 = BanditParameter("bandit1", np.array([[3, 4.0, 0], [3, 4, 0], [4, 4, 1]]))
    p2 = BanditParameter("bandit2", np.array([[0, 1], [1, 1], [1.0, 0]]))

    one_bandit_space = ParameterSpace([p1])
    x_test = np.array([[3.0, 4, 0], [4.0, 3, 0]])
    in_domain = one_bandit_space.check_points_in_domain(x_test)
    assert np.array_equal(in_domain, np.array([True, False]))

    two_bandits_space = ParameterSpace([p1, p2])
    x_test = np.array([[3.0, 4, 0, 0, 1], [4.0, 3, 0, 0, 1], [3, 4, 0, 1, 1]])
    in_domain = two_bandits_space.check_points_in_domain(x_test)
    assert np.array_equal(in_domain, np.array([True, False, True]))


def test_check_in_domain_with_bandit_parameter():
    mixed_space_with_bandit = ParameterSpace(
        [
            ContinuousParameter("c", 1.0, 5.0),
            DiscreteParameter("d", [0, 1, 2]),
            CategoricalParameter("cat", OneHotEncoding(["blue", "red"])),
            BanditParameter("bandit", np.array([[0, 1], [1, 1], [1.0, 0]])),
        ]
    )
    x_test = np.array([[1.5, 0, 1.0, 0.0, 0, 1], [1.5, 0, 1.0, 0.0, 0.0, 0.0]])
    in_domain = mixed_space_with_bandit.check_points_in_domain(x_test)
    assert np.array_equal(in_domain, np.array([True, False]))


def test_check_in_domain_fails(space_2d):
    x_test = np.array([[1.5, 6.0, 7.0], [1.5, 2.0, 7.0]])
    with pytest.raises(ValueError):
        space_2d.check_points_in_domain(x_test)


def test_two_information_source_parameters_fail():
    with pytest.raises(ValueError):
        ParameterSpace([InformationSourceParameter(2), InformationSourceParameter(2)])


def test_get_parameter_by_name(space_2d):
    param = space_2d.get_parameter_by_name("c")
    assert param.max == 5.0


def test_get_parameter_name_fails_with_wrong_name(space_2d):
    with pytest.raises(ValueError):
        space_2d.get_parameter_by_name("invalid_name")


def test_duplicate_parameter_names_fail():
    p1 = ContinuousParameter("c", 1.0, 5.0)
    p2 = ContinuousParameter("c", 1.0, 6.0)

    with pytest.raises(ValueError):
        ParameterSpace([p1, p2])


def test_get_bounds(space_3d_mixed):
    assert space_3d_mixed.get_bounds() == [(1.0, 5.0), (1.0, 3.0), (0, 1), (0, 1)]


class MockRandom:
    """Mock the numpy random class to deterministic test stochastic functions.

    Use like:

    >>> @mock.patch('numpy.random', MockRandom())
    >>> def test_something():
    >>>     np.random.uniform(0, 1, 10)  # call on mock object
    >>>     ...
    """

    @classmethod
    def uniform(cls, low, high, size):
        return np.linspace(low, high - 10e-8, np.product(size)).reshape(size)

    @classmethod
    def randint(cls, low, high, size):
        return cls.uniform(low, high, size).astype(int)


@mock.patch("numpy.random", MockRandom())
def test_sample_uniform(space_3d_mixed):
    X = space_3d_mixed.sample_uniform(90)
    assert_array_equal(np.histogram(X[:, 0], 9)[0], np.repeat(10, 9))
    assert_array_equal(np.bincount(X[:, 1].astype(int)), [0, 30, 30, 30])
    assert_array_equal(np.bincount(X[:, 2].astype(int)), [45, 45])
    assert_array_equal(np.bincount(X[:, 3].astype(int)), [45, 45])
