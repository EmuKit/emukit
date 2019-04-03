from emukit.core.acquisition import Acquisition

import pytest
import numpy as np


@pytest.fixture(scope="module")
def simple_square_acquisition():
    class SimpleSquareAcquisition(Acquisition):
        def __init__(self):
            pass

        def evaluate(self, x):
            y = - x[:, 0]**2 + np.sum(x[:, 1:], axis=1) + 1
            return np.atleast_2d(y).T

        @property
        def has_gradients(self):
            return False

    return SimpleSquareAcquisition()
