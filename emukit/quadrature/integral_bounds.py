from typing import Union

import numpy as np


class IntegralBounds():
    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> None:
        """
        Defines the parameter space by specifying the integration bounds

        :param lower_bounds: Lower bounds of the integral, shape (1, input_dim)
        :param upper_bounds: Upper bounds of the integral, shape (1, input_dim)
        """
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds


    def check_in_domain(self, x: np.ndarray) -> np.ndarray:
        """
        Checks if the points in x lie between the min and max allowed values
        :param x: locations (n_points, input_dim)
        :return: a boolean array (n_points,) indicating whether each point is in domain
        """
        return np.all([np.all(self.lower_bounds < x, axis=1), np.all(self.upper_bounds > x, axis=1)], axis=0)

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds
