import GPy
import numpy as np

from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.parameter_space import ParameterSpace
from emukit.core.continuous_parameter import ContinuousParameter

from emukit.core.loop import UserFunctionWrapper, FixedIterationsStoppingCondition

from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper


def f(x):
    return x**2


def test_loop():
    n_iterations = 5

    x_init = np.random.rand(5, 1)
    y_init = np.random.rand(5, 1)

    # Make GPy model
    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = GPyModelWrapper(gpy_model)

    space = ParameterSpace([ContinuousParameter('x', 0, 1)])
    acquisition = ExpectedImprovement(model)

    # Make loop and collect points
    bo = BayesianOptimizationLoop(model=model, space=space, acquisition=acquisition, X_init=x_init, Y_init=y_init)
    bo.run_loop(UserFunctionWrapper(f), FixedIterationsStoppingCondition(n_iterations))

    # Check we got the correct number of points
    assert bo.loop_state.X.shape[0] == n_iterations + 5
