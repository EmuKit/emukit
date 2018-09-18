import mock

import numpy as np

from emukit.core.loop import LoopState, FixedIterationsStoppingCondition, FixedIntervalUpdater, Sequential, UserFunctionWrapper
from emukit.core.acquisition import Acquisition
from emukit.core.optimization import AcquisitionOptimizer
from emukit.core.interfaces import IModel


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
    # Value should be result of acquitision optimization
    assert(np.equal(np.array([[0.]]), next_points[0]))


def test_user_function_wrapper():
    def function_test(x):
        return x[:, 0:1]**2 + x[:, 1:2]**2

    user_function = UserFunctionWrapper(function_test)
    results = user_function.evaluate(np.random.rand(10, 2))

    assert len(results) == 10, "A different number of results were expected"
    for res in results:
        assert res.X.ndim == 1, "X results are expected to be 1 dimensional"
        assert res.Y.ndim == 1, "Y results are expected to be 1 dimensional"
