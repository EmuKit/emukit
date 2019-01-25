import mock
import numpy as np
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
