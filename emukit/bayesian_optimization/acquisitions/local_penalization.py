from typing import Tuple

import numpy as np
from scipy.stats import norm

from ...core.acquisition import Acquisition
from ...core.interfaces import IModel


class LocalPenalization(Acquisition):
    """
    Penalization function used in the local penalization method. It is the log of the following function:

    .. math::
        h(x) = \sum_i{\log\\left(\Phi\\left(\\frac{\|x - x_i\|_{2} - r_i}{s_i}\\right)\\right)}

    where :math:`x_i` are the points already in the batch
    """
    def __init__(self, model: IModel):
        """
        :param model: Model
        """

        self.x_batch = None
        self.radius = None
        self.scale = None
        self.model = model

    @property
    def has_gradients(self) -> bool:
        return True

    def update_batches(self, x_batch: np.ndarray, lipschitz_constant: float, f_min: float):
        """
        Updates the batches internally and pre-computes the parameters of the penalization function
        """
        self.x_batch = x_batch
        if x_batch is not None:
            self._compute_parameters(x_batch, lipschitz_constant, f_min)
        else:
            self.x_batch = None
            self.radius = None
            self.scale = None

    def _compute_parameters(self, x_batch, lipschitz_constant, f_min):
        """
        Pre-computes the parameters of a penalization function
        """
        mean, variance = self.model.predict(x_batch)

        # Limit how small the variance can be to avoid numerical issues
        epsilon = 1e-10
        variance = np.maximum(epsilon, variance)

        std_deviation = np.sqrt(variance)

        # Calculate function parameters
        radius = (mean - f_min) / lipschitz_constant
        scale = std_deviation / lipschitz_constant
        self.radius = radius.flatten()
        self.scale = scale.flatten()
        self.x_batch = x_batch

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the penalization function value
        """

        if self.x_batch is None:
            return np.ones((x.shape[0], 1))

        distances = _distance_calculation(x, self.x_batch)
        normalized_distance = (distances - self.radius) / self.scale
        return norm.logcdf(normalized_distance).sum(axis=1, keepdims=True)

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the penalization function value and gradients with respect to x
        """

        if self.x_batch is None:
            return np.ones((x.shape[0], 1)), np.zeros(x.shape)

        distances, d_dist_dx = _distance_with_gradient(x, self.x_batch)
        normalized_distance = (distances - self.radius) / self.scale
        h_func = norm.cdf(normalized_distance)
        d_value_dx = 0.5 * (1 / h_func[:, :, None]) \
                     * norm.pdf(normalized_distance)[:, :, None] \
                     * d_dist_dx / self.scale[None, :, None]
        return norm.logcdf(normalized_distance).sum(1, keepdims=True), d_value_dx.sum(1)


def _distance_calculation(x_1, x_2):
    dx = x_1[:, None, :] - x_2[None, :, :]
    return np.sqrt(np.square(dx).sum(-1))


def _distance_with_gradient(x_1, x_2):
    distances = _distance_calculation(x_1, x_2)
    inv_distance = np.where(distances != 0., 1 / distances, 0)
    dx = x_1[:, None, :] - x_2[None, :, :]
    d_dist_dx = 2 * dx * inv_distance[:, :, None]
    return distances, d_dist_dx
