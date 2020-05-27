import numpy as np
import pytest

from emukit.core import ContinuousParameter, CategoricalParameter, ParameterSpace
from emukit.core import OneHotEncoding
from emukit.core.optimization import ContextManager

@pytest.fixture
def space():
    return ParameterSpace([ContinuousParameter('x1', 0, 15), ContinuousParameter('x2', 0, 15)])

@pytest.fixture
def context():
    return {'x1': 0.3}


def test_context_manager_idxs(space, context):
    context_manager = ContextManager(space, context)
    assert context_manager.non_context_idxs == [1]
    assert context_manager.context_idxs == [0]
    assert context_manager.context_values == [0.3]


def test_context_manager_expand_vector(space, context):
    context_manager = ContextManager(space, context)
    x = np.array([[0.5]])
    x_expanded = context_manager.expand_vector(x)
    assert x_expanded.ndim == 2
    assert np.array_equal(x_expanded, np.array([[0.3, 0.5]]))


@pytest.fixture
def catg_space():
    return ParameterSpace([ContinuousParameter('x1', 0, 15), 
                           CategoricalParameter('x2', OneHotEncoding([0, 1, 2, 3])),
                           CategoricalParameter('x3', OneHotEncoding([1, 2, 3, 4, 5])),
                           ContinuousParameter('x4', -2, 3)]) 

@pytest.fixture
def catg_context():
    return {'x2': 0, 'x3': 3}

def test_context_manager_catg_expand_vector(catg_space, catg_context):
    context_manager = ContextManager(catg_space, catg_context)
    x = np.array([[4., -1], [3, 0.]])
    print(context_manager.non_context_idxs)
    print(context_manager.context_idxs)

    x_expanded = context_manager.expand_vector(x)
    assert context_manager.space.dimensionality == 11
    assert x_expanded.shape == (2, context_manager.space.dimensionality)
    assert np.array_equal(x_expanded, np.array([[4., 1, 0, 0, 0, 0, 0, 1, 0, 0, -1.],[3, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0.]]))
    
