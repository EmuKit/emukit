import numpy as np
from numpy.testing import assert_array_equal

from emukit.core import ContinuousParameter
from emukit.core import InformationSourceParameter
from emukit.core import ParameterSpace
from emukit.core.optimization import AcquisitionOptimizer
from emukit.core.optimization import MultiSourceAcquisitionOptimizer


def test_multi_source_acquisition_optimizer(simple_square_acquisition):
    space = ParameterSpace([ContinuousParameter('x', 0, 1),
                            InformationSourceParameter(2)])
    single_optimizer = AcquisitionOptimizer(space)
    optimizer = MultiSourceAcquisitionOptimizer(single_optimizer, space)

    opt_x, opt_val = optimizer.optimize(simple_square_acquisition)
    assert_array_equal(opt_x, np.array([[0., 1.]]))
    assert_array_equal(opt_val, np.array([[2.]]))
