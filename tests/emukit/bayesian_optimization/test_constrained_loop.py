import GPy
import numpy as np

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import UnknownConstraintBayesianOptimizationLoop
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.loop import UserFunctionWrapper, FixedIterationsStoppingCondition
from emukit.model_wrappers import GPyModelWrapper


def f(x):
    return x**2, x - 0.5


def test_loop():
    n_iterations = 5

    x_init = np.random.rand(5, 1)
    y_init, y_constraint_init = f(x_init)

    # Make GPy objective model
    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = GPyModelWrapper(gpy_model)

    # Make GPy constraint model
    gpy_constraint_model = GPy.models.GPRegression(x_init, y_init)
    constraint_model = GPyModelWrapper(gpy_constraint_model)

    space = ParameterSpace([ContinuousParameter('x', 0, 1)])
    acquisition = ExpectedImprovement(model)

    # Make loop and collect points
    bo = UnknownConstraintBayesianOptimizationLoop(model_objective=model, space=space, acquisition=acquisition,
                                                   model_constraint=constraint_model)
    bo.run_loop(UserFunctionWrapper(f, extra_output_names=['Y_constraint']),
                FixedIterationsStoppingCondition(n_iterations))

    # Check we got the correct number of points
    assert bo.loop_state.X.shape[0] == n_iterations + 5
