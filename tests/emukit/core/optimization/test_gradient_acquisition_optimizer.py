import pytest
import numpy as np
from numpy.testing import assert_array_equal

from emukit.core import ContinuousParameter, CategoricalParameter, OneHotEncoding
from emukit.core import ParameterSpace
from emukit.core.optimization import GradientAcquisitionOptimizer


def test_gradient_acquisition_optimizer(simple_square_acquisition):
    space = ParameterSpace([ContinuousParameter('x', 0, 1)])
    optimizer = GradientAcquisitionOptimizer(space)
    opt_x, opt_val = optimizer.optimize(simple_square_acquisition)
    assert_array_equal(opt_x, np.array([[0.]]))
    assert_array_equal(opt_val, np.array([[1.]]))

def test_gradient_acquisition_optimizer_categorical(simple_square_acquisition):
    space = ParameterSpace([
        ContinuousParameter('x', 0, 1),
        CategoricalParameter('y', OneHotEncoding(['A', 'B']))
    ])
    optimizer = GradientAcquisitionOptimizer(space)
    context = {'y': 'B'}
    opt_x, opt_val = optimizer.optimize(simple_square_acquisition, context)
    assert_array_equal(opt_x, np.array([[0., 0., 1.]]))
    assert_array_equal(opt_val, np.array([[2.]]))
