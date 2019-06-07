import numpy as np

from emukit.core.loop import FixedIterationsStoppingCondition, UserFunctionWrapper
from emukit.examples.fabolas import FabolasLoop
from emukit.experimental_design.model_free.latin_design import LatinDesign


def fmin_fabolas(func, space, s_min, s_max, num_iters, n_init: int = 20):
    initial_design = LatinDesign(space)

    grid = initial_design.get_samples(n_init)
    X_init = np.zeros([n_init, grid.shape[1] + 1])
    Y_init = np.zeros([n_init, 1])
    cost_init = np.zeros([n_init])

    subsets = np.array([s_max // 2 ** i for i in range(2, 10)])
    idx = np.where(subsets < s_min)[0]
    subsets[idx] = s_min

    for it in range(n_init):
        func_val, cost = func(x=grid[it], s=subsets[it % len(subsets)])

        X_init[it] = np.concatenate((grid[it], np.array([subsets[it % len(subsets)]])))
        Y_init[it] = func_val
        cost_init[it] = cost

    def wrapper(x):
        y, c = func(x[0, :-1], x[0, -1])

        return np.array([[y]]), np.array([[c]])

    loop = FabolasLoop(X_init=X_init, Y_init=Y_init, cost_init=cost_init, space=space, s_min=s_min, s_max=s_max)
    loop.run_loop(user_function=UserFunctionWrapper(wrapper),
                  stopping_condition=FixedIterationsStoppingCondition(num_iters - n_init))

    return loop.loop_state
