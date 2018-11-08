from emukit.benchmarking.benchmarker import Benchmarker
from emukit.benchmarking.metrics import MinimumObservedValueMetric
from emukit.examples.enums import AcquisitionType, ModelType
from emukit.examples.optimization_loops import create_bayesian_optimization_loop
from emukit.examples.single_objective_bayesian_optimization import GPBayesianOptimization
from emukit.test_functions.branin import branin_function


def test_benchmarker():
    branin_fcn, parameter_space = branin_function()

    loops = [
        lambda x, y: create_bayesian_optimization_loop(x, y, parameter_space, AcquisitionType.EI, ModelType.RandomForest),
        lambda x, y: GPBayesianOptimization(parameter_space.parameters, x, y, acquisition_type=AcquisitionType.EI,
                                           noiseless=True)]

    n_repeats = 1
    n_initial_data = 5
    n_iterations = 10

    metrics = [MinimumObservedValueMetric()]

    benchmarkers = Benchmarker(loops, branin_fcn, parameter_space, metrics=metrics)
    benchmark_results = benchmarkers.run_benchmark(n_iterations=n_iterations, n_initial_data=n_initial_data,
                                                   n_repeats=n_repeats)

    assert benchmark_results.shape == (2, 1)
