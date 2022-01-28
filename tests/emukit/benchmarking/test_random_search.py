import numpy as np

from emukit.benchmarking.loop_benchmarking.random_search import RandomSearch
from emukit.core.loop import UserFunctionWrapper
from emukit.test_functions import branin_function


def test_random_search():
    np.random.seed(42)
    branin_fcn, parameter_space = branin_function()

    rs = RandomSearch(parameter_space)
    rs.run_loop(branin_fcn, 5)

    assert len(rs.loop_state.Y) == 5


def test_random_search_with_init_data():
    np.random.seed(42)
    branin_fcn, parameter_space = branin_function()

    branin_fcn_with_cost = lambda x: (branin_fcn(x), np.zeros((x.shape[0], 1)))

    # Ensure function returns a value for cost
    wrapped_fcn = UserFunctionWrapper(branin_fcn_with_cost, extra_output_names=["cost"])

    x_init = parameter_space.sample_uniform(5)
    y_init = branin_fcn(x_init)
    cost_init = np.ones([5, 1])

    rs = RandomSearch(parameter_space, x_init=x_init, y_init=y_init, cost_init=cost_init)
    rs.run_loop(wrapped_fcn, 5)

    assert len(rs.loop_state.Y) == 10
    assert len(rs.loop_state.X) == 10
    assert len(rs.loop_state.cost) == 10
