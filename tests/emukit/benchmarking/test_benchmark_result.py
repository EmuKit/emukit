import numpy as np
import pytest
from numpy.testing import assert_array_equal

from emukit.benchmarking.loop_benchmarking.benchmark_result import BenchmarkResult


def test_benchmark_result_initialization():
    br = BenchmarkResult(['loop1', 'loop2'], 5, ['mae', 'mse'])

    for loop in ['loop1', 'loop2']:
        for metric in ['mae', 'mse']:
            metric_data = br.extract_metric_as_array(loop, metric)

            assert metric_data.size == 0
            assert metric_data.shape == (5, 0)


def test_benchmark_result_extract_invalid_names():
    br = BenchmarkResult(['loop1', 'loop2'], 5, ['mae', 'mse'])

    with pytest.raises(KeyError):
        br.extract_metric_as_array('invalid loop', 'mse')

    with pytest.raises(KeyError):
        br.extract_metric_as_array('loop1', 'invalid metric')


def test_benchmark_result_add_data():
    br = BenchmarkResult(['loop1'], 2, ['mae', 'mse'])

    br.add_results('loop1', 0, 'mae', np.array([1, 2, 3]))
    br.add_results('loop1', 0, 'mse', np.array([1, 2]))
    br.add_results('loop1', 1, 'mae', np.array([4, 5, 6]))
    br.add_results('loop1', 1, 'mse', np.array([1, 2]))

    mae_data = br.extract_metric_as_array('loop1', 'mae')

    assert_array_equal(mae_data, np.array([[1, 2, 3], [4, 5, 6]]))

    mse_data = br.extract_metric_as_array('loop1', 'mse')

    assert_array_equal(mse_data, np.array([[1, 2], [1, 2]]))
