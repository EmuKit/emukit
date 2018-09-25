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
