import numpy as np

from emukit.core.continuous_parameter import ContinuousParameter
from emukit.examples.gp_bayesian_optimization.unknown_constraint_bayesian_optimization import GPBayesianOptimization
from emukit.core.loop import UserFunctionResult

def f(x):
    return x**2

def fc(x):
    return 2*x

def test_loop():
    n_iterations = 5
    x_init = np.random.rand(5, 1)
    y_init = np.random.rand(5, 1)
    yc_init = np.random.rand(5, 1)
    x = ContinuousParameter('x', 0, 1)
    bo = GPBayesianOptimization(variables_list=[x], X=x_init, Y=y_init, Yc=yc_init, batch_size=1)
    results = None
    l=0
    for _ in range(n_iterations):
    	X_new = bo.get_next_points(())
    	Y_new = f(X_new)
    	Yc_new = fc(X_new)
    	results = [UserFunctionResult(X_new[0], Y_new[0], Yc_new[0])]
    # Check we got the correct number of points
    assert bo.loop_state.X.shape[0] == n_iterations
