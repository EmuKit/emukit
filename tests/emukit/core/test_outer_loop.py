import GPy
import mock
import numpy as np
import pytest

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.interfaces import IModel
from emukit.core.loop import (
    CandidatePointCalculator,
    FixedIntervalUpdater,
    ModelUpdater,
    OuterLoop,
    SequentialPointCalculator,
    StoppingCondition,
    UserFunction,
    UserFunctionResult,
)
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.model_wrappers import GPyModelWrapper


@pytest.fixture
def mock_updater():
    updater = mock.create_autospec(ModelUpdater)
    updater.update.return_value = None

    return updater


@pytest.fixture
def mock_next_point_calculator():
    next_point = np.array([[0]])

    next_point_calculator = mock.create_autospec(CandidatePointCalculator)
    next_point_calculator.compute_next_points.return_value = next_point
    next_point_calculator.return_value.acquisition = mock.PropertyMock(None)

    return next_point_calculator


@pytest.fixture
def mock_user_function():
    user_function_results = [UserFunctionResult(np.array([0]), np.array([0]))]

    user_function = mock.create_autospec(UserFunction)
    user_function.evaluate.return_value = user_function_results

    return user_function


def test_outer_loop(mock_next_point_calculator, mock_updater, mock_user_function):
    """ Example of automatic outer loop """

    stopping_condition = mock.create_autospec(StoppingCondition)
    stopping_condition.should_stop.side_effect = [False, False, True]

    loop = OuterLoop(mock_next_point_calculator, mock_updater)
    loop.run_loop(mock_user_function, stopping_condition)

    assert (loop.loop_state.iteration == 2)
    assert (np.array_equal(loop.loop_state.X, np.array([[0], [0]])))


def test_outer_loop_model_update(mock_next_point_calculator, mock_user_function):
    """ Checks the model has the correct number of data points """

    class MockModelUpdater(ModelUpdater):
        def __init__(self, model):
            self.model = model

        def update(self, loop_state):
            self.model.set_data(loop_state.X, loop_state.Y)

    class MockModel(IModel):
        @property
        def X(self):
            return self._X

        @property
        def Y(self):
            return self._Y

        def predict(self, x):
            pass

        def set_data(self, x, y):
            self._X = x
            self._Y = y

        def optimize(self):
            pass

    mock_model = MockModel()
    model_updater = MockModelUpdater(mock_model)

    loop = OuterLoop(mock_next_point_calculator, model_updater)
    loop.run_loop(mock_user_function, 2)

    # Check update was last called with a loop state with all the collected data points
    assert mock_model.X.shape[0] == 2
    assert mock_model.Y.shape[0] == 2


def test_accept_non_wrapped_function(mock_next_point_calculator, mock_updater):
    stopping_condition = mock.create_autospec(StoppingCondition)
    stopping_condition.should_stop.side_effect = [False, False, True]

    def user_function(x):
        return np.array([[0]])

    loop = OuterLoop(mock_next_point_calculator, mock_updater)
    loop.run_loop(user_function, stopping_condition)

    assert (loop.loop_state.iteration == 2)
    assert (np.array_equal(loop.loop_state.X, np.array([[0], [0]])))


def test_default_condition(mock_next_point_calculator, mock_updater, mock_user_function):
    n_iter = 10

    loop = OuterLoop(mock_next_point_calculator, mock_updater)
    loop.run_loop(mock_user_function, n_iter)

    assert (loop.loop_state.iteration == n_iter)


def test_condition_invalid_type(mock_next_point_calculator, mock_updater, mock_user_function):
    invalid_n_iter = 10.0
    loop = OuterLoop(mock_next_point_calculator, mock_updater)

    with pytest.raises(ValueError):
        loop.run_loop(mock_user_function, invalid_n_iter)


def test_iteration_end_event():
    space = ParameterSpace([ContinuousParameter('x', 0, 1)])

    def user_function(x):
        return x

    x_test = np.linspace(0, 1)[:, None]
    y_test = user_function(x_test)

    x_init = np.linspace(0, 1, 5)[:, None]
    y_init = user_function(x_init)

    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = GPyModelWrapper(gpy_model)

    mse = []

    def compute_mse(self, loop_state):
        mse.append(np.mean(np.square(model.predict(x_test)[0] - y_test)))

    loop_state = create_loop_state(x_init, y_init)

    acquisition = ModelVariance(model)
    acquisition_optimizer = GradientAcquisitionOptimizer(space)
    candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)
    model_updater = FixedIntervalUpdater(model)

    loop = OuterLoop(candidate_point_calculator, model_updater, loop_state)
    loop.iteration_end_event.append(compute_mse)
    loop.run_loop(user_function, 5)

    assert len(mse) == 5
