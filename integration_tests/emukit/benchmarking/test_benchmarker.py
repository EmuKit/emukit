import GPy
import pytest

import emukit.test_functions
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.benchmarking.benchmarker import Benchmarker
from emukit.benchmarking.metrics import MinimumObservedValueMetric
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.model_wrappers import GPyModelWrapper


@pytest.fixture
def loops():
    space = ParameterSpace([ContinuousParameter('x', 0, 1)])

    def make_loop(x_init, y_init):
        gpy_model = GPy.models.GPRegression(x_init, y_init)
        model = GPyModelWrapper(gpy_model)
        return BayesianOptimizationLoop(model, space)

    return [('GP', make_loop)]


def test_benchmarker_runs(loops):
    test_fcn, parameter_space = emukit.test_functions.forrester_function()
    benchmark = Benchmarker(loops, test_fcn, parameter_space, [MinimumObservedValueMetric()])
    results = benchmark.run_benchmark()


def test_non_unique_metric_names_fail(loops):
    test_fcn, parameter_space = emukit.test_functions.forrester_function()
    with pytest.raises(ValueError):
        Benchmarker(loops, test_fcn, parameter_space, [MinimumObservedValueMetric('x'),
                                                       MinimumObservedValueMetric('x')])
