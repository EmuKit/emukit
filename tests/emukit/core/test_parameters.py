import numpy as np
import pytest

from emukit.core import ContinuousParameter
from emukit.core import DiscreteParameter, DiscreteInformationSourceParameter
from emukit.core import NominalInformationSourceParameter, OrdinalInformationSourceParameter


def test_continuous_parameter():
    param = ContinuousParameter('x', 1, 10)
    assert param.name == 'x'
    assert param.check_in_domain(np.array([1, 3]))
    assert param.check_in_domain(np.array([[1], [3]]))
    assert param.check_in_domain(3)
    assert not param.check_in_domain(0.9)
    assert not param.check_in_domain(0)

    with pytest.raises(ValueError):  # too many columns
        param.check_in_domain(np.array([[1, 0], [0, 2]]))
    with pytest.raises(ValueError):  # not a 1d/2d array
        param.check_in_domain(np.array([[[1]]]))


def test_discrete_parameter():
    param = DiscreteParameter('x', [0, 1, 2])
    assert param.name == 'x'
    assert param.check_in_domain(np.array([0, 1])) is True
    assert param.check_in_domain(np.array([3])) is False
    assert param.check_in_domain(np.array([[1], [0]])) is True
    assert param.check_in_domain([1, 0]) is True
    assert param.check_in_domain(1) is True
    assert param.check_in_domain(0.5) is False

    with pytest.raises(ValueError):  # too many columns
        param.check_in_domain(np.array([[1, 0], [0, 2]]))
    with pytest.raises(ValueError):  # not a 1d/2d array
        param.check_in_domain(np.array([[[1]]]))


def test_single_value_in_domain_discrete_parameter():
    param = DiscreteParameter('x', [0, 1, 2])
    assert param.check_in_domain(0) is True
    assert param.check_in_domain(3) is False


def test_information_source_parameter():
    param = NominalInformationSourceParameter(3)
    assert param.name == 'source'
    assert param.check_in_domain(np.array([[0, 1, 0]])) is True
    assert param.check_in_domain(np.array([[1, 0, 0]])) is True
    assert param.check_in_domain(np.array([[5, 5, 5]])) is False

    param = OrdinalInformationSourceParameter(5)
    assert param.name == 'source'
    assert param.check_in_domain(np.array([[1], [2]])) is True
    assert param.check_in_domain(np.array([[3]])) is True
    assert param.check_in_domain(np.array([[6]])) is False

    param = DiscreteInformationSourceParameter([0.1, 0.3, 0.9])
    assert param.name == 'source'
    assert param.check_in_domain(np.array([[0.1], [0.9]])) is True
    assert param.check_in_domain(np.array([0.3])) is True
    assert param.check_in_domain(np.array([[5]])) is False


def test_single_value_in_domain_information_source_parameter():
    param = DiscreteInformationSourceParameter(range(0, 5))
    assert param.check_in_domain(2) is True
    assert param.check_in_domain(7) is False
