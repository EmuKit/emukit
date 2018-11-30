import pytest
import numpy as np

from emukit.core import DiscreteParameter, InformationSourceParameter


def test_discrete_parameter():
    param = DiscreteParameter('x', [0, 1, 2])
    assert param.name == 'x'
    assert param.check_in_domain(np.array([0, 1])) is True
    assert param.check_in_domain(np.array([3])) is False


def test_single_value_in_domain_discrete_parameter():
    param = DiscreteParameter('x', [0, 1, 2])
    assert param.check_in_domain(0) is True
    assert param.check_in_domain(3) is False


def test_information_source_parameter():
    param = InformationSourceParameter(5)
    assert param.name == 'source'
    assert param.check_in_domain(np.array([0, 1])) is True
    assert param.check_in_domain(np.array([4])) is True
    assert param.check_in_domain(np.array([5])) is False


def test_single_value_in_domain_information_source_parameter():
    param = InformationSourceParameter(5)
    assert param.check_in_domain(2) is True
    assert param.check_in_domain(7) is False
