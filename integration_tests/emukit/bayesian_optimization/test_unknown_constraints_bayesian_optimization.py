import numpy as np

from emukit.core.continuous_parameter import ContinuousParameter
from emukit.core.loop import UserFunctionResult
from emukit.examples.gp_bayesian_optimization.unknown_constraint_bayesian_optimization import (
    UnknownConstraintGPBayesianOptimization,
)


def f(x):
    return x ** 2


def fc(x):
    return 2 * x


def test_loop():
    n_iterations = 5
    x_init = np.random.rand(5, 1)
    y_init = np.random.rand(5, 1)
    yc_init = np.random.rand(5, 1)
    x = ContinuousParameter('x', 0, 1)
    bo = UnknownConstraintGPBayesianOptimization(variables_list=[x], X=x_init, Y=y_init, Yc=yc_init, batch_size=1)
    results = None
    for _ in range(n_iterations + 1):
        X_new = bo.get_next_points(results)
        Y_new = f(X_new)
        Yc_new = fc(X_new)
        results = [UserFunctionResult(X_new[0], Y_new[0], Y_constraint=Yc_new[0])]
    # Check we got the correct number of points
    assert bo.loop_state.X.shape[0] == n_iterations + 5
