import mock
import pytest
import numpy as np

from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.constraints import IConstraint
from emukit.core.optimization.anchor_points_generator import ObjectiveAnchorPointsGenerator


def test_objective_anchor_point_generator():
    num_samples = 5
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition.evaluate.return_value = np.arange(num_samples)[:, None]

    space = mock.create_autospec(ParameterSpace)
    space.sample_uniform.return_value = np.arange(num_samples)[:, None]
    space.constraints = []

    generator = ObjectiveAnchorPointsGenerator(space, mock_acquisition, num_samples=num_samples)
    anchor_points = generator.get(1)

    # Check that the X that is picked corresponds to the highest acquisition value
    assert np.array_equal(anchor_points, np.array([[num_samples-1]]))


def test_constrained_objective_anchor_point_generator():
    num_samples = 5
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition.evaluate = lambda x: x

    space = mock.create_autospec(ParameterSpace)
    space.sample_uniform.return_value = np.arange(num_samples)[:, None]

    constraint = mock.create_autospec(IConstraint)
    constraint.evaluate.return_value = np.array([1, 1, 1, 0, 0])

    space.constraints = [constraint]

    generator = ObjectiveAnchorPointsGenerator(space, mock_acquisition, num_samples=num_samples)
    anchor_points = generator.get(1)

    # Check that the X that is picked corresponds to the highest acquisition value
    assert np.array_equal(anchor_points, np.array([[2]]))


def test_num_anchor_greater_than_num_samples_error():
    num_samples = 5
    num_anchor = 10
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition.evaluate = lambda x: x

    space = mock.create_autospec(ParameterSpace)
    space.sample_uniform.return_value = np.arange(num_samples)[:, None]

    constraint = mock.create_autospec(IConstraint)
    constraint.evaluate.return_value = np.array([1, 1, 1, 0, 0])

    space.constraints = [constraint]

    generator = ObjectiveAnchorPointsGenerator(space, mock_acquisition, num_samples=num_samples)

    with pytest.raises(ValueError):
        anchor_points = generator.get(num_anchor)
