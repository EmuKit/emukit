from typing import List

import numpy as np


class BenchmarkResult:
    def __init__(self, loop_names: List[str], n_repeats: int, metric_names: List[str]):
        """

        :param loop_names: List of loop names
        :param n_repeats: Number of random restarts in benchmarking
        :param metric_names: List of metric names
        """

        self.loop_names = loop_names
        self.n_repeats = n_repeats
        self.metric_names = metric_names

        self._results = dict()
        for loop_name in loop_names:
            self._results[loop_name] = dict()
            for metric_name in metric_names:
                self._results[loop_name][metric_name] = []
                for i in range(n_repeats):
                    self._results[loop_name][metric_name].append([])

    def add_results(self, loop_name: str, i_repeat: int, metric_name: str, metric_values: np.ndarray) -> None:
        """
        Add results for a specific loop, metric and repeat combination

        :param loop_name: Name of loop
        :param i_repeat: Index of repeat
        :param metric_name: Name of metric
        :param metric_values: Metric values to add
        """
        self._results[loop_name][metric_name][i_repeat] = metric_values.flatten()

    def extract_metric_as_array(self, loop_name: str, metric_name: str) -> np.ndarray:
        """
        Returns results over all repeats and iterations for a specific metric and loop name pair

        :param loop_name: Name of loop to return results for
        :param metric_name: Name of metric to extract
        :return: 2-d numpy array of shape (n_repeats x n_iterations)
        """
        return np.array(self._results[loop_name][metric_name])
