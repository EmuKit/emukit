import asyncio
import numpy as np
import GPy
import emukit

from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.loop import UserFunctionWrapper, UserFunctionResult
from emukit.core.loop.stopping_conditions import FixedIterationsStoppingCondition
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound

from emukit.test_functions.branin import (
    branin_function as _branin_function,
)

_branin, _ps = _branin_function()

async def a_cost(x: np.ndarray):
    # Cost function, defined arbitrarily
    t = sum(x)
    await asyncio.sleep(t)

async def a_objective(x: np.ndarray):
    # Objective function
    r = _branin(x)
    # await a_cost(x)
    return r

async def demo():
    # Configure
    _x = [7.5, 12.5]
    d = len(_x)
    x = np.array(_x).reshape((1, d))
    assert _ps.check_points_in_domain(x).all(), ("You configured a point outside the objective"
        f"function's domain: {x} is outside {_ps.get_bounds()}")
    # Execute
    print(f"Input: {x}")
    r = await a_objective(x)
    print(f"Output: {r}")

async def main():
    # Configure
    max_iter = 30
    n_init = 6
    batch_size = 6
    beta = 1  # tradeoff parameter for NCLB acq. opt.
    update_interval = 1  # how many results before running hyperparam. opt.
    # Build Bayesian optimization components
    space = _ps
    design = LatinDesign(space)
    X_init = design.get_samples(n_init)
    print(f"Initial design: X_init=\n{X_init}")
    input_coroutines = [a_objective(x.reshape((1,space.dimensionality))) for x in X_init]
    _Y_init = await asyncio.gather(*input_coroutines, return_exceptions=True)
    Y_init = np.concatenate(_Y_init)
    print(f"Initial design evaluated: Y_init=\n{Y_init}")
    model_gpy = GPRegression(X_init, Y_init)
    model_gpy.optimize()
    model_emukit = GPyModelWrapper(model_gpy)
    acquisition_function = NegativeLowerConfidenceBound(model=model_emukit, beta=beta)
    acquisition_optimizer = GradientAcquisitionOptimizer(space=space)
    bo_loop = BayesianOptimizationLoop(
        model = model_emukit,
        space = space,
        acquisition = acquisition_function,
        acquisition_optimizer = acquisition_optimizer,
        update_interval = update_interval,
        batch_size = batch_size,
    )
    # Run BO loop
    results = None
    n = bo_loop.model.X.shape[0]
    while n < max_iter:
        # TODO use a different acquisition function because currently X_batch is 5 identical sugg.
        # ^ only on occasion, apparently
        X_batch = bo_loop.get_next_points(results)
        coroutines = [a_objective(x.reshape((1, space.dimensionality))) for x in X_batch]
        # TODO update model as soon as any result is available
        # ^ as-is, only updates and makes new suggestions when all results come in
        _results = await asyncio.gather(*coroutines, return_exceptions=True)
        Y_batch = np.concatenate(_results)
        results = list(map(UserFunctionResult, X_batch, Y_batch))
        n = n + len(results)
    final_result = bo_loop.get_results()
    print(
        "Done!\n"
        f"Minimum found at location: {final_result.minimum_location}\n"
        f"With score: {final_result.minimum_value}"
        )

if __name__ == '__main__':
    asyncio.run(main())
