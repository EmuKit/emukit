import pytest
import numpy as np

from emukit.core import ContinuousParameter, ParameterSpace

@pytest.fixture
def space_2d():
    p1 = ContinuousParameter('c', 1.0, 5.0)
    p2 = ContinuousParameter('c', 1.0, 5.0)

    return ParameterSpace([p1, p2])


def test_parameter_space_has_all_parameters(space_2d):
    assert len(space_2d.parameters) == 2


def test_check_in_domain(space_2d):
    x_test = np.array([[1.5, 6.0], [1.5, 2.0]])
    in_domain = space_2d.check_points_in_domain(x_test)
    assert np.array_equal(in_domain, np.array([False, True]))


def test_check_in_domain_fails(space_2d):
    x_test = np.array([[1.5, 6.0, 7.0], [1.5, 2.0, 7.0]])
    with pytest.raises(ValueError):
        space_2d.check_points_in_domain(x_test)
