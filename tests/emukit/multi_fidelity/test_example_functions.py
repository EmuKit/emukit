"""
Runs all example functions
"""

import numpy as np
from emukit.multi_fidelity import example_functions


def test_forrester_high():
    x = np.random.rand(10, 1)
    y = example_functions.forrester_high(x)
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]


def test_forrester_low():
    x = np.random.rand(10, 1)
    y = example_functions.forrester_low(x)
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]


def test_borehole_low():
    x = np.random.rand(10, 8)
    y = example_functions.borehole_low(x)
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]


def test_borehole_high():
    x = np.random.rand(10, 8)
    y = example_functions.borehole_high(x)
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
