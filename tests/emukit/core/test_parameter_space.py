from unittest import mock
import itertools

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from emukit.core import ContinuousParameter, ParameterSpace, InformationSourceParameter, DiscreteParameter, \
    CategoricalParameter, OneHotEncoding


@pytest.fixture
def space_2d():
    p1 = ContinuousParameter('c', 1.0, 5.0)
    p2 = ContinuousParameter('d', 1.0, 6.0)

    return ParameterSpace([p1, p2])


@pytest.fixture
def space_3d_mixed():
    p1 = ContinuousParameter('c', 1.0, 5.0)
    p2 = DiscreteParameter('d', [1, 2, 3])
    p3 = CategoricalParameter('cat', OneHotEncoding(['Maine Coon', 'Siamese']))
    return ParameterSpace([p1, p2, p3])


def test_parameter_space_has_all_parameters(space_2d):
    assert len(space_2d.parameters) == 2


def test_check_in_domain(space_2d, space_3d_mixed):
    x_test = np.array([[1.5, 6.1], [1.5, 2.0]])
    in_domain = space_2d.check_points_in_domain(x_test)
    assert np.array_equal(in_domain, np.array([False, True]))

    x_mixed_test = np.array([[1.5, 0, 1, 1], [1.5, 1, 1., 0.]])
    in_domain = space_3d_mixed.check_points_in_domain(x_mixed_test)
    assert np.array_equal(in_domain, np.array([False, True]))


def test_check_in_domain_fails(space_2d):
    x_test = np.array([[1.5, 6.0, 7.0], [1.5, 2.0, 7.0]])
    with pytest.raises(ValueError):
        space_2d.check_points_in_domain(x_test)


def test_two_information_source_parameters_fail():
    with pytest.raises(ValueError):
        ParameterSpace([InformationSourceParameter(2), InformationSourceParameter(2)])


def test_get_parameter_by_name(space_2d):
    param = space_2d.get_parameter_by_name('c')
    assert param.max == 5.


def test_get_parameter_name_fails_with_wrong_name(space_2d):
    with pytest.raises(ValueError):
        space_2d.get_parameter_by_name('invalid_name')


def test_duplicate_parameter_names_fail():
    p1 = ContinuousParameter('c', 1.0, 5.0)
    p2 = ContinuousParameter('c', 1.0, 6.0)

    with pytest.raises(ValueError):
        ParameterSpace([p1, p2])


def test_get_bounds(space_3d_mixed):
    assert space_3d_mixed.get_bounds() == [(1., 5.), (1., 3.), (0, 1), (0, 1)]


class MockRandom:
    @staticmethod
    def uniform(low, high, size):
        return np.linspace(low, high - 10e-8, np.product(size)).reshape(size)


@mock.patch('numpy.random.uniform', MockRandom().uniform)
def test_sample_uniform(space_3d_mixed):
    X = space_3d_mixed.sample_uniform(90)
    assert_array_equal(np.histogram(X[:, 0], 9)[0], np.repeat(10, 9))
    assert_array_equal(np.bincount(X[:, 1].astype(int))[1:], [30, 30, 30])
    assert_array_equal(np.bincount(X[:, 2].astype(int))[1:], [45, 45])
    assert_array_equal(np.bincount(X[:, 3].astype(int))[1:], [45, 45])
