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
from emukit.core.acquisition import Acquisition
from emukit.core.optimization import AcquisitionOptimizer
from emukit.core.optimization import LocalSearchAcquisitionOptimizer
from emukit.core.optimization import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import RandomSearchAcquisitionOptimizer


class SimpleSquareAcquisition(Acquisition):
    def __init__(self):
        pass

    def evaluate(self, x):
        y = - x[:, 0]**2 + np.sum(x[:, 1:], axis=1) + 1
        return np.atleast_2d(y).T

    @property
    def has_gradients(self):
        return False


def test_acquisition_optimizer():
    space = ParameterSpace([ContinuousParameter('x', 0, 1)])
    acquisition = SimpleSquareAcquisition()
    optimizer = AcquisitionOptimizer(space)

    opt_x, opt_val = optimizer.optimize(acquisition)
    np.testing.assert_array_equal(opt_x, np.array([[0.]]))
    np.testing.assert_array_equal(opt_val, np.array([[1.]]))


def test_multi_source_acquisition_optimizer():
    space = ParameterSpace([ContinuousParameter('x', 0, 1),
                            InformationSourceParameter(2)])
    acquisition = SimpleSquareAcquisition()
    single_optimizer = AcquisitionOptimizer(space)
    optimizer = MultiSourceAcquisitionOptimizer(single_optimizer, space)

    opt_x, opt_val = optimizer.optimize(acquisition)
    np.testing.assert_array_equal(opt_x, np.array([[0., 1.]]))
    np.testing.assert_array_equal(opt_val, np.array([[2.]]))


def test_random_search_acquisition_optimizer():
    space = ParameterSpace([CategoricalParameter('x', OrdinalEncoding(np.arange(0, 100)))])
    acquisition = SimpleSquareAcquisition()
    optimizer = RandomSearchAcquisitionOptimizer(space, 1000)

    opt_x, opt_val = optimizer.optimize(acquisition)
    # ordinal encoding is as integers 1, 2, ...
    np.testing.assert_array_equal(opt_x, np.array([[1.]]))
    np.testing.assert_array_equal(opt_val, np.array([[0.]]))


def test_local_search_acquisition_optimizer():
    space = ParameterSpace([CategoricalParameter('x', OrdinalEncoding(np.arange(0, 100)))])
    acquisition = SimpleSquareAcquisition()
    optimizer = LocalSearchAcquisitionOptimizer(space, 1000, 3)

    opt_x, opt_val = optimizer.optimize(acquisition)
    # ordinal encoding is as integers 1, 2, ...
    np.testing.assert_array_equal(opt_x, np.array([[1.]]))
    np.testing.assert_array_equal(opt_val, np.array([[0.]]))


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

    neighbourhood = optimizer._neighbours_per_parameter(x)
    assert_equal(np.array([[0, 1, 0], [0, 0, 1]]), neighbourhood[0])
    assert_equal(np.array([[1], [3]]), neighbourhood[1])
    assert_equal(np.array([[2]]), neighbourhood[2])
    assert_equal(np.array([[1.2]]), neighbourhood[3])
    assert_almost_equal(np.array([[50.035281]]), neighbourhood[4])

    neighbours = optimizer._neighbours(x)
    assert_almost_equal(np.array([
        [0, 1, 0, 2., 3., 0.1, 50.],
        [0, 0, 1, 2., 3., 0.1, 50.],
        [1, 0, 0, 1., 3., 0.1, 50.],
        [1, 0, 0, 3., 3., 0.1, 50.],
        [1, 0, 0, 2., 2., 0.1, 50.],
        [1, 0, 0, 2., 3., 1.2, 50.],
        [1, 0, 0, 2., 3., 0.1, 50.00800314],
    ]), space.round(neighbours))
