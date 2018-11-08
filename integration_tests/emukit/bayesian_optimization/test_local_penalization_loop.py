import GPy

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.experimental_design import RandomDesign
from emukit.model_wrappers import GPyModelWrapper
from emukit.test_functions import branin_function


def test_local_penalization():
    branin_fcn, parameter_space = branin_function()
    random_design = RandomDesign(parameter_space)
    x_init = random_design.get_samples(10)

    y_init = branin_fcn(x_init)

    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = GPyModelWrapper(gpy_model)

    base_acquisition = ExpectedImprovement(model)

    batch_size = 10

    lp = BayesianOptimizationLoop(model, parameter_space, base_acquisition, batch_size)
    lp.run_loop(branin_fcn, 5)

    assert len(lp.loop_state.Y) == 60
