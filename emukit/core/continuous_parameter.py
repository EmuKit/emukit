# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Union, Tuple, List

import numpy as np

from .parameter import Parameter


class ContinuousParameter(Parameter):
    """
    A univariate continuous parameter with a domain defined in a range between two values
    """
    def __init__(self, name: str, min_value: float, max_value: float):
        """
        :param name: Name of parameter
        :param min_value: Minimum value the parameter is allowed to take
        :param max_value: Maximum value the parameter is allowed to take
        """
        super().__init__(name)
        self.min = min_value
        self.max = max_value

    def __str__(self):
        return f"<ContinuousParameter: {self.name} {self.bounds}>"

    def __repr__(self):
        return f"ContinuousParameter({self.name}, {self.min}, {self.max})"

    def check_in_domain(self, x: Union[np.ndarray, float]) -> bool:
        """
        Checks if all the points in x lie between the min and max allowed values

        :param x:    1d numpy array of points to check
                  or 2d numpy array with shape (n_points, 1) of points to check
                  or float of single point to check
        :return: A boolean value which indicates whether all points lie in the domain
        """
        if isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape[1] == 1:
                x = x.ravel()
            elif x.ndim > 1:
                raise ValueError("Expected x shape (n,) or (n, 1), actual is {}".format(x.shape))
        return np.all([self.min <= x, x <= self.max])

    @property
    def bounds(self) -> List[Tuple]:
        """
        Returns a list containing one tuple of minimum and maximum values parameter can take
        """
        return [(self.min, self.max)]

    def sample_uniform(self, point_count: int) -> np.ndarray:
        """
        Generates multiple uniformly distributed random parameter points.

        :param point_count: number of data points to generate.
        :returns: Generated points with shape (point_count, num_features)
        """
        return np.random.uniform(low=self.min, high=self.max, size=(point_count, 1))
