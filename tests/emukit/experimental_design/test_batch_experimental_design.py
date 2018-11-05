import GPy
import mock
import numpy as np
import pytest

from emukit.core import ParameterSpace, ContinuousParameter

from emukit.model_wrappers import GPyModelWrapper

from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization import AcquisitionOptimizer
from emukit.core.loop.candidate_point_calculators import GreedyBatchPointCalculator
from emukit.experimental_design.model_based.experimental_design_loop import ExperimentalDesignLoop


def test_batch_experimental_design_loop():

    class MockModel(GPyModelWrapper):
        def optimize(self):
            # speed up test by skipping the actual hyper-parameter optimization
            pass

    user_function = lambda x: x

    space = ParameterSpace([ContinuousParameter('x', 0, 3)])

    # Make model
    x_init = np.linspace(0, 3, 5)[:, None]
    y_init = user_function(x_init)
    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = MockModel(gpy_model)

    loop = ExperimentalDesignLoop(space, model, batch_size=5)
    loop.run_loop(user_function, 5)

    assert(loop.loop_state.iteration == 5)
    assert(loop.loop_state.X.shape[0] == 30)


def test_batch_point_calculator():
    class MockModel(IModel):
        def __init__(self):
            self._X = np.zeros((1, 1))
            self._Y = np.zeros((1, 1))

        def update_data(self, X, Y):
            self._X = X
            self._Y = Y

        def predict(self, X):
            return np.zeros((X.shape[0], 1)), np.zeros((X.shape[0], 1))

        @property
        def X(self):
            return self._X

        @property
        def Y(self):
            return self._Y

    acquisition = mock.create_autospec(Acquisition)
    acquisition_optimizer = mock.create_autospec(AcquisitionOptimizer)
    acquisition_optimizer.optimize.return_value = (np.zeros((1, 1)), 0)
    batch_size = 10

    calculator = GreedyBatchPointCalculator(MockModel(), acquisition, acquisition_optimizer, batch_size)

    loop_state = create_loop_state(np.zeros((1, 1)), np.zeros((1, 1)))
    next_points = calculator.compute_next_points(loop_state)
    assert next_points.shape[0] == batch_size


def test_zero_batch_size():
    space = ParameterSpace([ContinuousParameter('x', 0, 3)])

    # Make model
    model = mock.create_autospec(IModel)
    with pytest.raises(ValueError):
        ExperimentalDesignLoop(space, model, batch_size=0)


def test_non_integer_batch_size():
    class MockModel(IModel):
        def __init__(self):
            self._X = np.zeros((1, 1))
            self._Y = np.zeros((1, 1))

        def update_data(self, X, Y):
            self._X = X
            self._Y = Y

        def predict(self, X):
            return np.zeros((X.shape[0], 1)), np.zeros((X.shape[0], 1))

        @property
        def X(self):
            return self._X

        @property
        def Y(self):
            return self._Y

    space = ParameterSpace([ContinuousParameter('x', 0, 3)])

    # Make model
    model = MockModel()
    with pytest.raises(ValueError):
        ExperimentalDesignLoop(space, model, batch_size=3.5)
