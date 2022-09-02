import GPy
import numpy as np

from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.constraints import LinearInequalityConstraint, NonlinearInequalityConstraint
from emukit.model_wrappers import GPyModelWrapper
from emukit.test_functions import branin_function


def test_optimization_with_linear_constraint():
    branin_fcn, parameter_space = branin_function()
    x_init = parameter_space.sample_uniform(10)
    y_init = branin_fcn(x_init)

    A = np.array([[1.0, 1.0]])
    b_lower = np.array([-5])
    b_upper = np.array([5])
    parameter_space.constraints = [LinearInequalityConstraint(A, b_lower, b_upper)]

    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = GPyModelWrapper(gpy_model)

    lp = BayesianOptimizationLoop(parameter_space, model)
    lp.run_loop(branin_fcn, 5)

    assert True
