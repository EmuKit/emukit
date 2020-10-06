import numpy as np

from emukit.core.continuous_parameter import ContinuousParameter
from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization


def f(x):
    return x**2


def test_loop():
    n_iterations = 5

    x_init = np.random.rand(5, 1)
    y_init = np.random.rand(5, 1)
    x = ContinuousParameter('x', 0, 1)
    bo = GPBayesianOptimization(variables_list=[x], X=x_init, Y=y_init)
    bo.run_optimization(f, n_iterations)

    # Check we got the correct number of points
    assert bo.loop_state.X.shape[0] == n_iterations + 5
    assert bo.suggest_new_locations().shape == (1, 1)
