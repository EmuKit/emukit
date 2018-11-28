import GPy
import numpy as np

from emukit.core import ParameterSpace, ContinuousParameter, CategoricalParameter

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.experimental_design import RandomDesign
from emukit.model_wrappers import GPyModelWrapper


def test_categorical_variables():
    np.random.seed(123)

    def objective(x):
        return np.array(np.sum(x, axis=1).reshape(-1, 1))

    parameter_space = ParameterSpace([
        ContinuousParameter('real_param', 0.0, 1.0),
        CategoricalParameter('categorical_param', np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    ])

    random_design = RandomDesign(parameter_space)
    x_init = random_design.get_samples(10)

    assert x_init.shape == (10, 4)
    assert np.all(np.logical_or(x_init[:, 1:3] == 0.0, x_init[:, 1:3] == 1.0))

    y_init = objective(x_init)

    gpy_model = GPy.models.GPRegression(x_init, y_init)
    gpy_model.Gaussian_noise.fix(1)
    model = GPyModelWrapper(gpy_model)

    acquisition = ExpectedImprovement(model)

    loop = BayesianOptimizationLoop(model, parameter_space, acquisition)
    loop.run_loop(objective, 5)

    assert len(loop.loop_state.Y) == 15
    assert np.all(np.logical_or(loop.loop_state.X[:, 1:3] == 0.0, loop.loop_state.X[:, 1:3] == 1.0))
