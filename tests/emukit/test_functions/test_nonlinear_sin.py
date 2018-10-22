import numpy as np

from emukit.test_functions.non_linear_sin import nonlinear_sin_high, nonlinear_sin_low


def test_non_linear_high_return_shape():
    """
    Test output dimension is 2d
    """
    x = np.array([[0.2], [0.4]])
    assert nonlinear_sin_high(x, 0).shape == (2, 1)


def test_non_linear_low_return_shape():
    """
    Test output dimension is 2d
    """
    x = np.array([[0.2], [0.4]])
    assert nonlinear_sin_low(x, 0).shape == (2, 1)
