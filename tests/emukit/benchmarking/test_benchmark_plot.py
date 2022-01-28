import numpy as np
import pytest

from emukit.benchmarking.loop_benchmarking.benchmark_plot import BenchmarkPlot
from emukit.benchmarking.loop_benchmarking.benchmark_result import BenchmarkResult


@pytest.fixture
def benchmark_result():
    br = BenchmarkResult(["loop1", "loop2"], 2, ["mae", "mse"])

    br.add_results("loop1", 0, "mae", np.array([1, 2, 3]))
    br.add_results("loop1", 0, "mse", np.array([1, 2]))
    br.add_results("loop1", 1, "mae", np.array([4, 5, 6]))
    br.add_results("loop1", 1, "mse", np.array([1, 2]))

    br.add_results("loop2", 0, "mae", np.array([1]))
    br.add_results("loop2", 0, "mse", np.array([2]))
    br.add_results("loop2", 1, "mae", np.array([4]))
    br.add_results("loop2", 1, "mse", np.array([1]))

    return br


def test_benchmark_plot_metrics_to_plot(benchmark_result):
    bp = BenchmarkPlot(benchmark_result)

    assert bp.metrics_to_plot == benchmark_result.metric_names

    bp = BenchmarkPlot(benchmark_result, metrics_to_plot=["mae"])

    assert bp.metrics_to_plot == ["mae"]

    with pytest.raises(ValueError):
        BenchmarkPlot(benchmark_result, metrics_to_plot=["invalid metric"])


def test_benchmark_plot_x_axis_metric_name(benchmark_result):
    bp = BenchmarkPlot(benchmark_result)

    assert bp.x_label == "Iteration"
    assert bp.x_axis is None
    assert bp.metrics_to_plot == benchmark_result.metric_names

    bp = BenchmarkPlot(benchmark_result, x_axis_metric_name="mae")

    assert bp.x_label == "mae"
    assert bp.x_axis == "mae"
    assert "mae" not in bp.metrics_to_plot

    with pytest.raises(ValueError):
        BenchmarkPlot(benchmark_result, x_axis_metric_name="invalid metric")


def test_benchmark_plot_make_plot(benchmark_result):
    bp = BenchmarkPlot(benchmark_result)
    bp.make_plot()

    assert bp.fig_handle is not None
    assert len(bp.fig_handle.get_axes()) == len(benchmark_result.metric_names)
