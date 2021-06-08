import numpy as np

from emukit.core.loop import FixedIterationsStoppingCondition, UserFunctionWrapper, LoopState
from emukit.core.parameter_space import ParameterSpace
from emukit.examples.fabolas import FabolasLoop
from emukit.core.initial_designs.latin_design import LatinDesign


def fmin_fabolas(func, space: ParameterSpace, s_min: float, s_max: float, n_iters: int,
                 n_init: int = 20, marginalize_hypers: bool = True) -> LoopState:
    """
    Simple interface for Fabolas which optimizes the hyperparameters of machine learning algorithms
    by reasoning across training data set subsets. For further details see:

    Fast Bayesian hyperparameter optimization on large datasets
    A. Klein and S. Falkner and S. Bartels and P. Hennig and F. Hutter
    Electronic Journal of Statistics (2017)

    :param func: objective function which gets a hyperparameter configuration x and training dataset size s as input,
    and return the validation error and the runtime after training x on s datapoints.
    :param space: input space
    :param s_min: minimum training dataset size (linear scale)
    :param s_max: maximum training dataset size (linear scale)
    :param n_iters: number of iterations
    :param n_init: number of initial design points (needs to be smaller than num_iters)
    :param marginalize_hypers: determines whether to use a MAP estimate or to marginalize over the GP hyperparameters

    :return: LoopState with all evaluated data points
    """
    initial_design = LatinDesign(space)

    grid = initial_design.get_samples(n_init)
    X_init = np.zeros([n_init, grid.shape[1] + 1])
    Y_init = np.zeros([n_init, 1])
    cost_init = np.zeros([n_init])

    subsets = np.array([s_max // 2 ** i for i in range(2, 10)])[::-1]
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

    loop = FabolasLoop(X_init=X_init, Y_init=Y_init, cost_init=cost_init, space=space, s_min=s_min,
                       s_max=s_max, marginalize_hypers=marginalize_hypers)
    loop.run_loop(user_function=UserFunctionWrapper(wrapper, extra_output_names=["cost"]),
                  stopping_condition=FixedIterationsStoppingCondition(n_iters - n_init))

    return loop.loop_state
