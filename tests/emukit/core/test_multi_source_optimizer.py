import mock
import numpy as np
import pytest

from emukit.core import ContinuousParameter, OrdinalInformationSourceParameter, ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.loop import LoopState, SequentialPointCalculator
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer


@pytest.fixture
def multi_source_optimizer():
    mock_acquisition_optimizer = mock.create_autospec(GradientAcquisitionOptimizer)
    mock_acquisition_optimizer.optimize.return_value = (np.array([[0.]]), None)
    space = ParameterSpace([ContinuousParameter('x', 0, 1), OrdinalInformationSourceParameter(2)])
    return MultiSourceAcquisitionOptimizer(mock_acquisition_optimizer, space)


def test_multi_source_optimizer_returns_one_point(multi_source_optimizer):
    # SequentialPointCalculator should just return result of the acquisition optimizer
    next_points, _ = multi_source_optimizer.optimize(mock.create_autospec(Acquisition))

    # "SequentialPointCalculator" should only ever return 1 value
    assert(len(next_points) == 1)


def test_multi_source_optimizer_returns_2d_array(multi_source_optimizer):
    # SequentialPointCalculator should just return result of the acquisition optimizer
    next_points, _ = multi_source_optimizer.optimize(mock.create_autospec(Acquisition))

    # Check output is 2d
    assert next_points.ndim == 2


def test_multi_source_optimizer_returns_correct_result(multi_source_optimizer):
    # SequentialPointCalculator should just return result of the acquisition optimizer
    next_points = multi_source_optimizer.optimize(mock.create_autospec(Acquisition))
    # Value should be result of acquisition optimization
    assert np.equal(np.array([0.]), next_points[0])


def test_multi_source_sequential_with_context():
    # Check that we can fix a non-information source parameter with context
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition.has_gradients = False
    mock_acquisition.evaluate = lambda x: np.sum(x**2, axis=1)[:, None]
    space = ParameterSpace([ContinuousParameter('x', 0, 1),
                            ContinuousParameter('y', 0, 1),
                            OrdinalInformationSourceParameter(2)])
    acquisition_optimizer = GradientAcquisitionOptimizer(space)
    multi_source_acquisition_optimizer = MultiSourceAcquisitionOptimizer(acquisition_optimizer, space)

    loop_state_mock = mock.create_autospec(LoopState)
    seq = SequentialPointCalculator(mock_acquisition, multi_source_acquisition_optimizer)
    next_points = seq.compute_next_points(loop_state_mock, context={'x': 0.25})

    # "SequentialPointCalculator" should only ever return 1 value
    assert(len(next_points) == 1)
    # Context value should be what we set
    assert np.isclose(next_points[0, 0], 0.25)


def test_multi_source_sequential_with_source_context():
    # Check that we can fix a non-information source parameter with context
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition.has_gradients = False
    mock_acquisition.evaluate = lambda x: np.sum(x**2, axis=1)[:, None]
    space = ParameterSpace([ContinuousParameter('x', 0, 1),
                            ContinuousParameter('y', 0, 1),
                            OrdinalInformationSourceParameter(2)])
    acquisition_optimizer = GradientAcquisitionOptimizer(space)
    multi_source_acquisition_optimizer = MultiSourceAcquisitionOptimizer(acquisition_optimizer, space)

    loop_state_mock = mock.create_autospec(LoopState)
    seq = SequentialPointCalculator(mock_acquisition, multi_source_acquisition_optimizer)
    next_points = seq.compute_next_points(loop_state_mock, context={'source': 1.0})

    # "SequentialPointCalculator" should only ever return 1 value
    assert(len(next_points) == 1)
    # Context value should be what we set
    assert np.isclose(next_points[0, 1], 1.)
