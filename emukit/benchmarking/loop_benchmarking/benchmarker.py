import logging
import numpy as np
from typing import Callable, List, Tuple, Union

from ...core import ParameterSpace
from ...core.loop import LoopState, UserFunction, UserFunctionWrapper
from ...experimental_design import RandomDesign
from ...experimental_design.model_free.base import ModelFreeDesignBase
from .benchmark_result import BenchmarkResult
from .metrics import Metric

_log = logging.getLogger(__name__)


class Benchmarker:
    def __init__(self, loops_with_names: List[Tuple[str, Callable]], test_function: Union[Callable, UserFunction],
                 parameter_space: ParameterSpace, metrics: List[Metric], initial_design: ModelFreeDesignBase=None):
        """
        :param loops_with_names: A list of tuples where the first entry is the name of the loop and the second is a
                                 function that takes in initial x and y training data and returns a loop to be
                                 benchmarked
        :param test_function: The function to benchmark the loop against
        :param parameter_space: Parameter space describing the input domain of the function to be benchmarked against
        :param metrics: List of metric objects that assess the performance of the loop at every iteration
        :param initial_design: An object that returns a set of samples in the input domain that are used as the initial
                               data set
        """

        self.loop_names = [loop[0] for loop in loops_with_names]
        self.loops = [loop[1] for loop in loops_with_names]

        if isinstance(test_function, UserFunction):
            self.test_function = test_function
        else:
            self.test_function = UserFunctionWrapper(test_function)
        self.parameter_space = parameter_space

        if initial_design is None:
            initial_design = RandomDesign(parameter_space)
        self.initial_design = initial_design
        self.metrics = metrics
        self.metric_names = [metric.name for metric in metrics]

        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError('Names of metrics are not unique')

    def run_benchmark(self, n_initial_data: int=10, n_iterations: int=10, n_repeats: int=10) -> BenchmarkResult:
        """
        Runs the benchmarking. For each initial data set, every loop is created and run for the specified number of
        iterations and the results are collected.

        :param n_initial_data: Number of points in the initial data set
        :param n_iterations: Number of iterations to run the loop for
        :param n_repeats: Number of times to run each loop with a different initial data set
        :return: An instance of BenchmarkResult that contains all the tracked metrics for each loop
        """
        result = BenchmarkResult(self.loop_names, n_repeats, self.metric_names)
        for j in range(n_repeats):
            initial_loop_state = self._create_initial_loop_state(n_initial_data)
            for i, (loop, loop_name) in enumerate(zip(self.loops, self.loop_names)):
                _log.info('Benchmarking loop ' + str(i) + ' for repeat ' + str(j))

                this_loop = loop(initial_loop_state.X, initial_loop_state.Y)
                this_loop.loop_state.metrics = dict()
                self._subscribe_metrics_to_loop_events(this_loop)

                this_loop.run_loop(self.test_function, n_iterations)

                for metric_name, metric_value in this_loop.loop_state.metrics.items():
                    result.add_results(loop_name, j, metric_name, metric_value)
        return result

    def _subscribe_metrics_to_loop_events(self, outer_loop):
        """
        Subscribe metric calls to events on outer loop object
        """
        if self.metrics is not None:
            for metric in self.metrics:
                metric.reset()

                def update_metric(loop, loop_state):
                    value = metric.evaluate(loop, loop_state)
                    _add_value_to_metrics_dict(loop_state, value, metric.name)

                # Subscribe to events
                outer_loop.loop_start_event.append(update_metric)
                outer_loop.iteration_end_event.append(update_metric)

    def _create_initial_loop_state(self, n_initial_data):
        x_init = self.initial_design.get_samples(n_initial_data)
        results = self.test_function.evaluate(x_init)
        return LoopState(results)


def _add_value_to_metrics_dict(loop_state, value, key_name):
    """
    Add new metric evaluation to dictionary in loop state using the metric name as the key in the dictionary
    """

    if key_name in loop_state.metrics:
        # Array already exists - append new value
        loop_state.metrics[key_name] = np.concatenate([loop_state.metrics[key_name], [value]], axis=0)
    else:
        # Initialise array
        loop_state.metrics[key_name] = np.array([value])