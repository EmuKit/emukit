from time import time

import numpy as np

from emukit.core.loop import LoopState, OuterLoop


class Metric:
    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> None:
        pass

    def reset(self):
        pass


class MeanSquaredErrorMetric(Metric):
    """
    Mean squared error metric stored in loop state metric dictionary with key "mean_squared_error".
    """
    def __init__(self, x_test: np.ndarray, y_test: np.ndarray, name: str='mean_squared_error'):
        """
        :param x_test: Input locations of test data
        :param y_test: Test targets
        """
        self.x_test = x_test
        self.y_test = y_test
        self.name = name

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> None:
        """
        Calculate and store mean squared error

        :param loop: Outer loop
        :param loop_state: Object containing history of the loop that we add results to
        """
        # Calculate mean squared error
        predictions = loop.model_updater.model.predict(self.x_test)[0]
        mse = np.mean(np.square(self.y_test - predictions), axis=0)
        # Add to metrics dictionary in loop state
        _add_value_to_metrics_dict(loop_state, mse, self.name)


class MinimumObservedValueMetric(Metric):
    """
    The result is stored in the "metrics" dictionary in the loop state with the key "minimum_observed_value"
    """
    def __init__(self, name: str='minimum_observed_value'):
        self.name = name

    def evaluate(self, loop, loop_state) -> None:
        """
        Evaluates minimum observed value

        :param loop: Outer loop
        :param loop_state: Object containing history of the loop that we add results to
        """
        y_min = np.min(loop_state.Y, axis=0)
        # Add to metrics dictionary in loop state
        _add_value_to_metrics_dict(loop_state, y_min, self.name)


class TimeMetric(Metric):
    def __init__(self, name: str='time'):
        self.start_time = None
        self.name = name

    def reset(self):
        self.start_time = time()

    def evaluate(self, loop: OuterLoop, loop_state: LoopState):
        time_since_start = time() - self.start_time
        # Add to metrics dictionary in loop state
        _add_value_to_metrics_dict(loop_state, time_since_start, self.name)


def _add_value_to_metrics_dict(loop_state, value, key_name):
    if key_name in loop_state.metrics:
        loop_state.metrics[key_name] = np.concatenate([loop_state.metrics[key_name], value], axis=0)
    else:
        loop_state.metrics[key_name] = np.array([value])
