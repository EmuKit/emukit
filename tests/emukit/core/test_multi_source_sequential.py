import mock
import pytest

import numpy as np

from emukit.core import InformationSourceParameter, ContinuousParameter, ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.loop import MultiSourceSequential, LoopState
from emukit.core.optimization import AcquisitionOptimizer


@pytest.fixture
def multi_source_sequential():
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition_optimizer = mock.create_autospec(AcquisitionOptimizer)
    mock_acquisition_optimizer.optimize.return_value = (np.array([[0.]]), None)
    space = ParameterSpace([ContinuousParameter('x', 0, 1), InformationSourceParameter(2)])
    return MultiSourceSequential(mock_acquisition, mock_acquisition_optimizer, space)


def test_multi_source_sequential_returns_one_point(multi_source_sequential):
    # Sequential should just return result of the acquisition optimizer
    loop_state_mock = mock.create_autospec(LoopState)
    next_points = multi_source_sequential.compute_next_points(loop_state_mock)

    # "Sequential" should only ever return 1 value
    assert(len(next_points) == 1)


def test_multi_source_sequential_returns_2d_array(multi_source_sequential):
    # Sequential should just return result of the acquisition optimizer
    loop_state_mock = mock.create_autospec(LoopState)
    next_points = multi_source_sequential.compute_next_points(loop_state_mock)

    # Check output is 2d
    assert next_points.ndim == 2


def test_multi_source_sequential_returns_correct_result(multi_source_sequential):
    # Sequential should just return result of the acquisition optimizer
    loop_state_mock = mock.create_autospec(LoopState)
    next_points = multi_source_sequential.compute_next_points(loop_state_mock)
    # Value should be result of acquisition optimization
    assert np.equal(np.array([0.]), next_points[0])


def test_multi_source_sequential_with_context():
    # Check that we can fix a non-information source parameter with context
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition.has_gradients = False
    mock_acquisition.evaluate = lambda x: np.sum(x**2, axis=1)[:, None]
    space = ParameterSpace([ContinuousParameter('x', 0, 1), ContinuousParameter('y', 0, 1),  InformationSourceParameter(2)])
    acquisition_optimizer = AcquisitionOptimizer(space)

    loop_state_mock = mock.create_autospec(LoopState)
    seq = MultiSourceSequential(mock_acquisition, acquisition_optimizer, space)
    next_points = seq.compute_next_points(loop_state_mock, context={'x': 0.25})

    # "Sequential" should only ever return 1 value
    assert(len(next_points) == 1)
    # Context value should be what we set
    assert np.isclose(next_points[0, 0], 0.25)


def test_multi_source_sequential_with_source_context():
    # Check that we can fix the information source parameter with context
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition.has_gradients = False
    mock_acquisition.evaluate = lambda x: np.sum(x**2, axis=1)[:, None]
    space = ParameterSpace([ContinuousParameter('x', 0, 1), InformationSourceParameter(2)])
    acquisition_optimizer = AcquisitionOptimizer(space)

    loop_state_mock = mock.create_autospec(LoopState)
    seq = MultiSourceSequential(mock_acquisition, acquisition_optimizer, space)
    next_points = seq.compute_next_points(loop_state_mock, context={'source': 1.0})

    # "Sequential" should only ever return 1 value
    assert(len(next_points) == 1)
    # Context value should be what we set
    assert np.isclose(next_points[0, 1], 1.)