import numpy as np
import pytest

from emukit.core import (
    BanditParameter,
    ContinuousParameter,
    DiscreteParameter,
    InformationSourceParameter,
)


def test_bandit_parameter():
    domain = np.array([[1,1],[1,2],[2,2]])
    param = BanditParameter('x',domain)
    assert param.name == 'x'
    assert not param.check_in_domain(np.array([1, 3]))
    assert param.check_in_domain(np.array([1,2]))
    assert all(param.check_in_domain(np.array([[1,1], [1,2]])))
    assert (param.round(np.array([[1,1.2]])) == [1,1]).all()
    assert all([isinstance(sp, DiscreteParameter) for sp in param.parameters])
    assert all(param.check_in_domain(param.sample_uniform(10)))

    with pytest.raises(ValueError):  # too many columns
        param.check_in_domain(np.array([[1, 0, 0], [0, 2, 0]]))
    with pytest.raises(ValueError):  # not a 1d/2d array
        param.check_in_domain(np.array([[[1]]]))
    with pytest.raises(ValueError):  # wrong type
        param.check_in_domain((1,1))
    with pytest.raises(ValueError):  # wrong type
        param.check_in_domain(1)


def test_bandit_parameter_str_repr():
    domain = np.array([[1,2,3],[2,3,4]])
    for subparam_names in [['foo','bar'], None]:
        param = BanditParameter('foobar', domain.T, subparam_names)
        _ = str(param)
        _ = repr(param)


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


def test_continuous_parameter_str_repr():
    param = ContinuousParameter('x', 1, 10)
    _ = str(param)
    _ = repr(param)


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


def test_discrete_parameter_str_repr():
    param = DiscreteParameter('x', [0, 1, 2])
    _ = str(param)
    _ = repr(param)


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
