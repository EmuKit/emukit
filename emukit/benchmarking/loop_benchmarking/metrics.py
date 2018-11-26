from time import time

import numpy as np

from ...core.loop import LoopState, OuterLoop


class Metric:
    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> None:
        pass

    def reset(self) -> None:
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

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        """
        Calculate and store mean squared error

        :param loop: Outer loop
        :param loop_state: Object containing history of the loop that we add results to
        """
        # Calculate mean squared error
        predictions = loop.model_updaters[0].model.predict(self.x_test)[0]
        mse = np.mean(np.square(self.y_test - predictions), axis=0)
        return mse


class MinimumObservedValueMetric(Metric):
    """
    The result is stored in the "metrics" dictionary in the loop state with the key "minimum_observed_value"
    """
    def __init__(self, name: str='minimum_observed_value'):
        self.name = name

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        """
        Evaluates minimum observed value

        :param loop: Outer loop
        :param loop_state: Object containing history of the loop that we add results to
        """
        y_min = np.min(loop_state.Y, axis=0)
        return y_min


class TimeMetric(Metric):
    """
    Time taken between each iteration of the loop
    """
    def __init__(self, name: str='time'):
        """
        :param name: Name of the metric. Defaults to "time"
        """
        self.start_time = None
        self.name = name

    def evaluate(self, loop: OuterLoop, loop_state: LoopState) -> np.ndarray:
        """
        Returns difference between time now and when the reset method was last called
        """
        time_since_start = time() - self.start_time
        # Add to metrics dictionary in loop state
        return np.array([time_since_start])

    def reset(self) -> None:
        """
        Resets the start time
        """
        self.start_time = time()
