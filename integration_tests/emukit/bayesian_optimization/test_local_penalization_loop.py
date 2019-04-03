import GPy
import numpy as np

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.model_wrappers import GPyModelWrapper
from emukit.test_functions import branin_function


def test_local_penalization():
    np.random.seed(123)
    branin_fcn, parameter_space = branin_function()
    x_init = parameter_space.sample_uniform(10)

    y_init = branin_fcn(x_init)

    gpy_model = GPy.models.GPRegression(x_init, y_init)
    gpy_model.Gaussian_noise.fix(1)
    model = GPyModelWrapper(gpy_model)

    base_acquisition = ExpectedImprovement(model)

    batch_size = 10
    update_interval = 1

    lp = BayesianOptimizationLoop(parameter_space, model, base_acquisition, update_interval, batch_size)
    lp.run_loop(branin_fcn, 5)

    assert len(lp.loop_state.Y) == 60
