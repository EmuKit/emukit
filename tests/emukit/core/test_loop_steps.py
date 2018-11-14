import mock
import numpy as np

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel
from emukit.core.loop import (FixedIntervalUpdater, FixedIterationsStoppingCondition, LoopState, Sequential,
                              UserFunctionWrapper)
from emukit.core.optimization import AcquisitionOptimizer


def test_fixed_iteration_stopping_condition():
    stopping_condition = FixedIterationsStoppingCondition(5)
    loop_state_mock = mock.create_autospec(LoopState)
    loop_state_mock.iteration = 0

    assert(stopping_condition.should_stop(loop_state_mock) is False)

    loop_state_mock = mock.create_autospec(LoopState)
    loop_state_mock.iteration = 5

    assert(stopping_condition.should_stop(loop_state_mock) is True)


def test_every_iteration_model_updater():
    mock_model = mock.create_autospec(IModel)
    mock_model.optimize.return_value(None)
    updater = FixedIntervalUpdater(mock_model, 1)

    loop_state_mock = mock.create_autospec(LoopState)
    loop_state_mock.iteration = 1
    loop_state_mock.X.return_value(np.random.rand(5, 1))
    loop_state_mock.Y.return_value(np.random.rand(5, 1))
    updater.update(loop_state_mock)
    mock_model.optimize.assert_called_once()


def test_every_iteration_model_updater_with_cost():
    """
    Tests that the model updater can use a different attribute from loop_state as the training targets
    """

    class MockModel(IModel):
        def optimize(self):
            pass
        
        def set_data(self, X: np.ndarray, Y: np.ndarray):
            self._X = X
            self._Y = Y

        @property
        def X(self):
            return self._X

        @property
        def Y(self):
            return self._Y

    mock_model = MockModel()
    updater = FixedIntervalUpdater(mock_model, 1, lambda loop_state: loop_state.cost)

    loop_state_mock = mock.create_autospec(LoopState)
    loop_state_mock.iteration = 1
    loop_state_mock.X.return_value(np.random.rand(5, 1))
    loop_state_mock.cost.return_value(np.random.rand(5, 1))

    cost = np.random.rand(5, 1)
    loop_state_mock.cost.return_value(cost)
    updater.update(loop_state_mock)
    assert np.array_equiv(mock_model.X, cost)


def test_sequential_evaluator():
    # Sequential should just return result of the acquisition optimizer
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition_optimizer = mock.create_autospec(AcquisitionOptimizer)
    mock_acquisition_optimizer.optimize.return_value = (np.array([[0.]]), None)
    loop_state_mock = mock.create_autospec(LoopState)
    seq = Sequential(mock_acquisition, mock_acquisition_optimizer)
    next_points = seq.compute_next_points(loop_state_mock)

    # "Sequential" should only ever return 1 value
    assert(len(next_points) == 1)
    # Value should be result of acquisition optimization
    assert(np.equal(np.array([[0.]]), next_points[0]))


def test_sequential_with_context():
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition.has_gradients = False
    mock_acquisition.evaluate = lambda x: np.sum(x**2, axis=1)[:, None]
    space = ParameterSpace([ContinuousParameter('x', 0, 1), ContinuousParameter('y', 0, 1)])
    acquisition_optimizer = AcquisitionOptimizer(space)

    loop_state_mock = mock.create_autospec(LoopState)
    seq = Sequential(mock_acquisition, acquisition_optimizer)
    next_points = seq.compute_next_points(loop_state_mock, context={'x': 0.25})

    # "Sequential" should only ever return 1 value
    assert(len(next_points) == 1)
    # Context value should be what we set
    assert np.isclose(next_points[0, 0], 0.25)


def test_sequential_with_all_parameters_fixed():
    mock_acquisition = mock.create_autospec(Acquisition)
    mock_acquisition.has_gradients = False
    mock_acquisition.evaluate = lambda x: np.sum(x**2, axis=1)[:, None]
    space = ParameterSpace([ContinuousParameter('x', 0, 1), ContinuousParameter('y', 0, 1)])
    acquisition_optimizer = AcquisitionOptimizer(space)

    loop_state_mock = mock.create_autospec(LoopState)
    seq = Sequential(mock_acquisition, acquisition_optimizer)
    next_points = seq.compute_next_points(loop_state_mock, context={'x': 0.25, 'y': 0.25})
    assert np.array_equiv(next_points, np.array([0.25, 0.25]))


def test_user_function_wrapper():
    def function_test(x):
        return x[:, 0:1]**2 + x[:, 1:2]**2

    user_function = UserFunctionWrapper(function_test)
    results = user_function.evaluate(np.random.rand(10, 2))

    assert len(results) == 10, "A different number of results were expected"
    for res in results:
        assert res.X.ndim == 1, "X results are expected to be 1 dimensional"
        assert res.Y.ndim == 1, "Y results are expected to be 1 dimensional"
