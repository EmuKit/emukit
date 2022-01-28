import GPy
import numpy as np

from emukit.bayesian_optimization.loops.cost_sensitive_bayesian_optimization_loop import (
    CostSensitiveBayesianOptimizationLoop,
)
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.model_wrappers import GPyModelWrapper


def test_cost_sensitive_bayesian_optimization_loop():
    space = ParameterSpace([ContinuousParameter("x", 0, 1)])

    x_init = np.random.rand(10, 1)

    def function_with_cost(x):
        return np.sin(x), x

    user_fcn = UserFunctionWrapper(function_with_cost, extra_output_names=["cost"])

    y_init, cost_init = function_with_cost(x_init)

    gpy_model_objective = GPy.models.GPRegression(x_init, y_init)
    gpy_model_cost = GPy.models.GPRegression(x_init, cost_init)

    model_objective = GPyModelWrapper(gpy_model_objective)
    model_cost = GPyModelWrapper(gpy_model_cost)

    loop = CostSensitiveBayesianOptimizationLoop(space, model_objective, model_cost)
    loop.run_loop(user_fcn, 10)

    assert loop.loop_state.X.shape[0] == 20
    assert loop.loop_state.cost.shape[0] == 20
