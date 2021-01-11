import mock
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from emukit.core import CategoricalParameter


def test_categorical_parameter(encoding):
    param = CategoricalParameter('v', encoding)
    assert param.name == 'v'
    assert param.dimension == 3
    assert len(param.model_parameters) == 3


def test_categorical_parameter_rounding(encoding):
    expected = np.array([[1, 2, 4], [2, 3, 5]])
    encoding.round = mock.MagicMock(return_value=expected)
    param = CategoricalParameter('v', encoding)

    assert_array_equal(param.round(np.ones((2, 3))), expected)


def test_categorical_parameter_check_in_domain(encoding):
    param = CategoricalParameter('v', encoding)

    assert param.check_in_domain(np.array([[1, 0, 0], [0, 0.5, 0]]))
    assert not param.check_in_domain(np.array([[2, 0, 0], [0, 0.5, 0]]))

    with pytest.raises(ValueError):  # wrong encoding dimension
        param.check_in_domain(np.array([[1, 0], [0, 0.5]]))
    with pytest.raises(ValueError):  # not a 2d array
        param.check_in_domain(np.array([1, 0, 0]))


def test_categorical_parameter_str_repr(encoding):
    param = CategoricalParameter('v', encoding)
    _ = str(param)
    _ = repr(param)
