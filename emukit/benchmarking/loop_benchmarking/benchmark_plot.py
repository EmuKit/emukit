from itertools import cycle
from typing import List

import numpy as np

from .benchmark_result import BenchmarkResult

try:
    import matplotlib.pyplot as plt
except ImportError:
    ImportError('matplotlib needs to be installed in order to use BenchmarkPlot')


class BenchmarkPlot:
    """
    Creates a plot comparing the results from the different loops used during benchmarking
    """
    def __init__(self, benchmark_results: BenchmarkResult, loop_colours: List=None, loop_line_styles: List[str]=None,
                 x_axis_metric_name: str=None, metrics_to_plot: List[str]=None):
        """
        :param benchmark_results: The output of a benchmark run
        :param loop_colours: Colours to use for each loop. Defaults to standard matplotlib colour palette
        :param loop_line_styles: Line style to use for each loop. Defaults to solid line style for all lines
        :param x_axis_metric_name: Which metric to use as the x axis in plots.
                                   None means it will be plotted against iteration number.
        :param metrics_to_plot: A list of metric names to plot. Defaults to all metrics apart from the one used as the
                                x axis.
        """
        self.benchmark_results = benchmark_results
        self.loop_names = benchmark_results.loop_names

        if loop_colours is None:
            self.loop_colours = _get_default_colours()
        else:
            self.loop_colours = loop_colours

        if loop_line_styles is None:
            self.loop_line_styles = ['-']
        else:
            self.loop_line_styles = loop_line_styles

        if metrics_to_plot is None:
            self.metrics_to_plot = self.benchmark_results.metric_names
        else:
            for metric_name in metrics_to_plot:
                if metric_name not in self.benchmark_results.metric_names:
                    raise ValueError(metric_name + ' not found in saved metrics from benchmark results.')
            self.metrics_to_plot = metrics_to_plot

        if x_axis_metric_name is not None:
            if x_axis_metric_name not in self.metrics_to_plot:
                raise ValueError('x_axis ' + x_axis_metric_name + ' is not a valid metric name')
            self.metrics_to_plot.remove(x_axis_metric_name)

        if x_axis_metric_name is None:
            self.x_label = 'Iteration'
        else:
            self.x_label = x_axis_metric_name

        self.fig_handle = None
        self.x_axis = x_axis_metric_name

    def make_plot(self) -> None:
        """
        Make one plot for each metric measured, comparing the different loop results against each other
        """

        n_metrics = len(self.metrics_to_plot)
        self.fig_handle, _ = plt.subplots(n_metrics, 1)

        for i, metric_name in enumerate(self.metrics_to_plot):
            # Initialise plot
            plt.subplot(n_metrics, 1, i + 1)
            plt.title(metric_name)

            colours = cycle(self.loop_colours)
            line_styles = cycle(self.loop_line_styles)
            min_x = np.inf
            max_x = -np.inf

            for j, loop_name in enumerate(self.loop_names):
                # Get all results for this metric
                metric = self.benchmark_results.extract_metric_as_array(loop_name, metric_name)

                # Get plot options
                colour = next(colours)
                line_style = next(line_styles)

                # Get data to plot
                mean, std = _get_metric_stats(metric)

                if self.x_axis is not None:
                    x = np.mean(self.benchmark_results.extract_metric_as_array(loop_name, self.x_axis), axis=0)
                else:
                    x = np.arange(0, mean.shape[0])

                # Save min/max of data to set the axis limits later
                min_x = np.min([np.min(x), min_x])
                max_x = np.max([np.max(x), max_x])

                # Plot
                plt.plot(x, mean, color=colour, linestyle=line_style)
                plt.xlabel(self.x_label)
                plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=colour)

            # Make legend
            plt.legend(self.loop_names)
            plt.tight_layout()

            plt.xlim(min_x, max_x)

    def save_plot(self, file_name: str) -> None:
        """
        Save plot to file

        :param file_name:
        """
        if self.fig_handle is None:
            raise ValueError('Please call make_plots method before saving to file')

        with open(file_name) as file:
            self.fig_handle.savefig(file)


def _get_metric_stats(metric):
    return metric.mean(axis=0), metric.std(axis=0)


def _get_default_colours():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']
