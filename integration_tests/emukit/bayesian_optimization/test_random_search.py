import numpy as np

from emukit.test_functions import branin_function
from emukit.examples.gp_bayesian_optimization.random_search import RandomSearch


def test_random_search():
    np.random.seed(42)
    branin_fcn, parameter_space = branin_function()

    rs = RandomSearch(parameter_space)
    rs.run_loop(branin_fcn, 5)

    assert len(rs.loop_state.Y) == 5
