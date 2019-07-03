import numpy as np
import pytest

from emukit.core import ContinuousParameter, ParameterSpace
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
