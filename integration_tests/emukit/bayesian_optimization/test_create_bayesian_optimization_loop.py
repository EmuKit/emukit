import numpy as np

from emukit.test_functions.forrester import forrester_function
from emukit.examples.gp_bayesian_optimization.optimization_loops import create_bayesian_optimization_loop
from emukit.examples.gp_bayesian_optimization.enums import AcquisitionType, ModelType


def test_loop_state():

    fcn, cs = forrester_function()
    n_init = 5

    x_init = np.random.rand(n_init, 1)
    y_init = np.random.rand(n_init, 1)
    c_init = np.random.rand(n_init, 1)
    bo = create_bayesian_optimization_loop(x_init=x_init, y_init=y_init,
                                           parameter_space=cs, acquisition_type=AcquisitionType.EI,
                                           model_type=ModelType.RandomForest, cost_init=c_init)

    assert bo.loop_state.X.shape[0] == n_init
    assert bo.loop_state.Y.shape[0] == n_init
    assert bo.loop_state.cost.shape[0] == n_init
