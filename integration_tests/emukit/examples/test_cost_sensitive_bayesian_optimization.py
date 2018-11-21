import GPy
import numpy as np

from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.examples.cost_sensitive_bayesian_optimization.cost_sensitive_bayesian_optimization_loop import CostSensitiveBayesianOptimizationLoop
from emukit.model_wrappers import GPyModelWrapper


def test_cost_sensitive_bayesian_optimization_loop():
    space = ParameterSpace([ContinuousParameter('x', 0, 1)])

    x_init = np.random.rand(10, 1)

    def function_with_cost(x):
        return np.sin(x), x

    user_fcn = UserFunctionWrapper(function_with_cost)

    y_init, cost_init = function_with_cost(x_init)

    gpy_model_objective = GPy.models.GPRegression(x_init, y_init)
    gpy_model_cost = GPy.models.GPRegression(x_init, cost_init)

    model_objective = GPyModelWrapper(gpy_model_objective)
    model_cost = GPyModelWrapper(gpy_model_cost)

    loop = CostSensitiveBayesianOptimizationLoop(model_objective, model_cost, space, x_init, y_init, cost_init)
    loop.run_loop(user_fcn, 10)

    assert loop.loop_state.X.shape[0] == 20
    assert loop.loop_state.cost.shape[0] == 20
