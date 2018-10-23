import numpy as np

from emukit.test_functions import borehole_function
from emukit.test_functions.borehole import borehole_low


def test_borehole_return_shape():
    """
    Test output dimension is 2d
    """
    borehole, _ = borehole_function()
    x = np.zeros((2, 8))
    assert borehole(x).ndim == 2
    assert borehole(x).shape == (2, 1)


def test_borehole_low_return_shape():
    x = np.zeros((2, 8))
    assert borehole_low(x, 0).shape == (2, 1)
