import pytest

import emukit.test_functions
from emukit.benchmarking.benchmarker import Benchmarker
from emukit.benchmarking.metrics import MinimumObservedValueMetric
from emukit.core import ContinuousParameter
from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization


@pytest.fixture
def loops():
    vars = [ContinuousParameter('x', 0, 1)]
    return [('GP', lambda X, Y: GPBayesianOptimization(vars, X, Y, False))]


def test_benchmarker_runs(loops):
    test_fcn, parameter_space = emukit.test_functions.forrester_function()
    benchmark = Benchmarker(loops, test_fcn, parameter_space, [MinimumObservedValueMetric()])
    results = benchmark.run_benchmark()


def test_non_unique_metric_names_fail(loops):
    test_fcn, parameter_space = emukit.test_functions.forrester_function()
    with pytest.raises(ValueError):
        Benchmarker(loops, test_fcn, parameter_space, [MinimumObservedValueMetric('x'),
                                                       MinimumObservedValueMetric('x')])
