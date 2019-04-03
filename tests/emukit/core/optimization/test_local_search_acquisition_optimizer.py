import pytest
import numpy as np
from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal

from emukit.core import CategoricalParameter, ContinuousParameter
from emukit.core import DiscreteParameter, InformationSourceParameter
from emukit.core import OneHotEncoding, OrdinalEncoding
from emukit.core import ParameterSpace, Parameter
from emukit.core.encodings import Encoding
from emukit.core.optimization import LocalSearchAcquisitionOptimizer


def test_local_search_acquisition_optimizer(simple_square_acquisition):
    space = ParameterSpace([CategoricalParameter('x', OrdinalEncoding(np.arange(0, 100)))])
    optimizer = LocalSearchAcquisitionOptimizer(space, 1000, 3)

    opt_x, opt_val = optimizer.optimize(simple_square_acquisition)
    # ordinal encoding is as integers 1, 2, ...
    np.testing.assert_array_equal(opt_x, np.array([[1.]]))
    np.testing.assert_array_equal(opt_val, np.array([[0.]]))

    class UnknownParameter(Parameter):
        def __init__(self, name: str):
            self.name = name
    space.parameters.append(UnknownParameter('y'))
    with pytest.raises(TypeError):
        optimizer.optimize(simple_square_acquisition)
    space.parameters.pop()

    class UnknownEncoding(Encoding):
        def __init__(self):
            super().__init__([1], [[1]])
    space.parameters.append(CategoricalParameter('y', UnknownEncoding()))
    with pytest.raises(TypeError):
        optimizer.optimize(simple_square_acquisition)
    space.parameters.pop()


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
        DiscreteParameter('f', [0.1, 1.2, 2.3]),
    ])
    x = np.array([1, 0, 0, 1.6, 2.9, 0.1, 50, 1.2])
    optimizer = LocalSearchAcquisitionOptimizer(space, 1000, 3, num_continuous=1)

    neighbourhood = optimizer._neighbours_per_parameter(x, space.parameters)
    assert_equal(np.array([[0, 1, 0], [0, 0, 1]]), neighbourhood[0])
    assert_equal(np.array([[1], [3]]), neighbourhood[1])
    assert_equal(np.array([[2]]), neighbourhood[2])
    assert_equal(np.array([[1.2]]), neighbourhood[3])
    assert_almost_equal(np.array([[53.5281047]]), neighbourhood[4])
    assert_equal(np.array([[0.1], [2.3]]), neighbourhood[5])

    neighbours = optimizer._neighbours(x, space.parameters)
    assert_almost_equal(np.array([
        [0, 1, 0, 2., 3., 0.1, 50., 1.2],
        [0, 0, 1, 2., 3., 0.1, 50., 1.2],
        [1, 0, 0, 1., 3., 0.1, 50., 1.2],
        [1, 0, 0, 3., 3., 0.1, 50., 1.2],
        [1, 0, 0, 2., 2., 0.1, 50., 1.2],
        [1, 0, 0, 2., 3., 1.2, 50., 1.2],
        [1, 0, 0, 2., 3., 0.1, 50.80031442, 1.2],
        [1, 0, 0, 2., 3., 0.1, 50., 0.1],
        [1, 0, 0, 2., 3., 0.1, 50., 2.3],
    ]), space.round(neighbours))
