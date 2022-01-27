import GPy
import mock
import numpy as np
import pytest

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.continuous_parameter import ContinuousParameter
from emukit.core.interfaces import IModel
from emukit.core.loop import FixedIterationsStoppingCondition, UserFunctionWrapper
from emukit.core.parameter_space import ParameterSpace
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper


def f(x):
    return x**2


@pytest.mark.parametrize("batch_size", [1, 3])
def test_loop(batch_size):
    n_init = 5
    n_iterations = 5

    x_init = np.random.rand(n_init, 1)
    y_init = np.random.rand(n_init, 1)

    # Make GPy model
    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = GPyModelWrapper(gpy_model)

    space = ParameterSpace([ContinuousParameter('x', 0, 1)])
    acquisition = ExpectedImprovement(model)

    # Make loop and collect points
    bo = BayesianOptimizationLoop(model=model, space=space, acquisition=acquisition, batch_size=batch_size)
    bo.run_loop(UserFunctionWrapper(f), FixedIterationsStoppingCondition(n_iterations))

    # Check we got the correct number of points
    assert bo.loop_state.X.shape[0] == n_iterations * batch_size + n_init

    # Check the obtained results
    results = bo.get_results()

    assert results.minimum_location.shape[0] == 1
    assert results.best_found_value_per_iteration.shape[0] == n_iterations * batch_size + n_init
