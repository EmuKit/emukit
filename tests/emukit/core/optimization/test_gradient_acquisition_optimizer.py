import numpy as np
from numpy.testing import assert_array_equal

from emukit.core import ContinuousParameter
from emukit.core import ParameterSpace
from emukit.core.optimization import AcquisitionOptimizer


def test_acquisition_optimizer(simple_square_acquisition):
    space = ParameterSpace([ContinuousParameter('x', 0, 1)])
    optimizer = AcquisitionOptimizer(space)

    opt_x, opt_val = optimizer.optimize(simple_square_acquisition)
    assert_array_equal(opt_x, np.array([[0.]]))
    assert_array_equal(opt_val, np.array([[1.]]))
