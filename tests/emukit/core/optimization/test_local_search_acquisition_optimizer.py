import numpy as np
from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal

from emukit.core import CategoricalParameter
from emukit.core import ContinuousParameter
from emukit.core import DiscreteParameter
from emukit.core import InformationSourceParameter
from emukit.core import OneHotEncoding
from emukit.core import OrdinalEncoding
from emukit.core import ParameterSpace
from emukit.core.optimization import LocalSearchAcquisitionOptimizer


def test_local_search_acquisition_optimizer(simple_square_acquisition):
    space = ParameterSpace([CategoricalParameter('x', OrdinalEncoding(np.arange(0, 100)))])
    optimizer = LocalSearchAcquisitionOptimizer(space, 1000, 3)

    opt_x, opt_val = optimizer.optimize(simple_square_acquisition)
    # ordinal encoding is as integers 1, 2, ...
    np.testing.assert_array_equal(opt_x, np.array([[1.]]))
    np.testing.assert_array_equal(opt_val, np.array([[0.]]))


def test_local_search_acquisition_optimizer_with_context(simple_square_acquisition):
    space = ParameterSpace([CategoricalParameter('x', OrdinalEncoding(np.arange(0, 100))),
                            InformationSourceParameter(10)])
    optimizer = LocalSearchAcquisitionOptimizer(space, 1000, 3)

    source_encoding = 1
    opt_x, opt_val = optimizer.optimize(simple_square_acquisition, {'source': source_encoding})
    np.testing.assert_array_equal(opt_x, np.array([[1., source_encoding]]))
    np.testing.assert_array_equal(opt_val, np.array([[0. + source_encoding]]))


def test_local_search_acquisition_optimizer_neighbours():
    np.random.seed(0)
    space = ParameterSpace([
        CategoricalParameter('a', OneHotEncoding([1, 2, 3])),
        CategoricalParameter('b', OrdinalEncoding([0.1, 1, 2])),
        CategoricalParameter('c', OrdinalEncoding([0.1, 1, 2])),
        DiscreteParameter('d', [0.1, 1.2, 2.3]),
        ContinuousParameter('e', 0, 100),
    ])
    x = np.array([1, 0, 0, 1.6, 2.9, 0.1, 50])
    optimizer = LocalSearchAcquisitionOptimizer(space, 1000, 3)

    neighbourhood = optimizer._neighbours_per_parameter(x, space.parameters)
    assert_equal(np.array([[0, 1, 0], [0, 0, 1]]), neighbourhood[0])
    assert_equal(np.array([[1], [3]]), neighbourhood[1])
    assert_equal(np.array([[2]]), neighbourhood[2])
    assert_equal(np.array([[1.2]]), neighbourhood[3])
    assert_almost_equal(np.array([[50.035281]]), neighbourhood[4])

    neighbours = optimizer._neighbours(x, space.parameters)
    assert_almost_equal(np.array([
        [0, 1, 0, 2., 3., 0.1, 50.],
        [0, 0, 1, 2., 3., 0.1, 50.],
        [1, 0, 0, 1., 3., 0.1, 50.],
        [1, 0, 0, 3., 3., 0.1, 50.],
        [1, 0, 0, 2., 2., 0.1, 50.],
        [1, 0, 0, 2., 3., 1.2, 50.],
        [1, 0, 0, 2., 3., 0.1, 50.00800314],
    ]), space.round(neighbours))
