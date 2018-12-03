from typing import Tuple

import numpy as np

from ...core.acquisition import Acquisition


class LogAcquisition(Acquisition):
    """
    Takes the log of an acquisition function.
    """

    def __init__(self, acquisition: Acquisition):
        """
        :param acquisition: Base acquisition function that is log transformed. This acquisition function must output
                            positive values only.
        """
        self.acquisition = acquisition

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Input location
        :return: log of original acquisition function at input location(s)
        """
        return np.log(self.acquisition.evaluate(x))

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return log of original acquisition with gradient

        :param x: Input location
        :return: Tuple of (log value, gradient of log value)
        """
        value, gradient = self.acquisition.evaluate_with_gradients(x)

        epsilon = 1e-10
        value = np.maximum(value, epsilon)
        log_gradient = 1 / value * gradient
        return np.log(value), log_gradient

    @property
    def has_gradients(self) -> bool:
        return self.acquisition.has_gradients
