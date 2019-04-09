import numpy as np
from numpy.testing import assert_array_equal

from emukit.core import CategoricalParameter
from emukit.core import InformationSourceParameter
from emukit.core import OrdinalEncoding
from emukit.core import ParameterSpace
from emukit.core.optimization import RandomSearchAcquisitionOptimizer


def test_random_search_acquisition_optimizer(simple_square_acquisition):
    space = ParameterSpace([CategoricalParameter('x', OrdinalEncoding(np.arange(0, 100)))])
    optimizer = RandomSearchAcquisitionOptimizer(space, 1000)

    opt_x, opt_val = optimizer.optimize(simple_square_acquisition)
    # ordinal encoding is as integers 1, 2, ...
    assert_array_equal(opt_x, np.array([[1.]]))
    assert_array_equal(opt_val, np.array([[0.]]))


def test_random_search_acquisition_optimizer_with_context(simple_square_acquisition):
    space = ParameterSpace([CategoricalParameter('x', OrdinalEncoding(np.arange(0, 100))),
                            InformationSourceParameter(10)])
    optimizer = RandomSearchAcquisitionOptimizer(space, 1000)

    source_encoding = 1
    opt_x, opt_val = optimizer.optimize(simple_square_acquisition, {'source': source_encoding})
    assert_array_equal(opt_x, np.array([[1., source_encoding]]))
    assert_array_equal(opt_val, np.array([[0. + source_encoding]]))
