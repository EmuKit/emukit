# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Iterable, Union, Tuple, List

import numpy as np

from .parameter import Parameter


class DiscreteParameter(Parameter):
    """
    A parameter that takes a discrete set of values where the order and spacing of values is important
    """
    def __init__(self, name: str, domain: Iterable):
        """
        :param name: Name of parameter
        :param domain: valid values the parameter can have
        """
        super().__init__(name)
        self.domain = domain

    def __str__(self):
        return f"<DiscreteParameter: {self.name} {self.bounds}>"

    def __repr__(self):
        return f"DiscreteParameter({self.name}, {self.domain})"

    def check_in_domain(self, x: Union[np.ndarray, Iterable, float]) -> bool:
        """
        Checks if the points in x are in the set of allowed values

        :param x: 1d numpy array of points to check
        :param x:    1d numpy array of points to check
                  or 2d numpy array with shape (n_points, 1) of points to check
                  or Iterable of points to check
                  or float of single point to check
        :return: A boolean indicating whether each point is in domain
        """
        if np.isscalar(x):
            x = [x]
        elif isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape[1] == 1:
                x = x.ravel()
            elif x.ndim > 1:
                raise ValueError("Expected x shape (n,) or (n, 1), actual is {}".format(x.shape))
        return set(x).issubset(set(self.domain))

    @property
    def bounds(self) -> List[Tuple]:
        """
        Returns a list containing one tuple of min and max values parameter can take
        """
        return [(min(self.domain), max(self.domain))]

    def round(self, x: np.ndarray) -> np.ndarray:
        """
        Rounds each row in x to represent a valid value for this discrete variable

        :param x: A 2d array Nx1 to be rounded
        :returns: An array Nx1 where each row represents a value from the domain
                  that is closest to the corresponding row in x
        """
        if x.ndim != 2:
            raise ValueError("Expected 2d array, got " + str(x.ndim))

        if x.shape[1] != 1:
            raise ValueError("Expected single column array, got {}".format(x.shape[1]))

        x_rounded = []
        for row in x:
            value = row[0]
            rounded_value = min(self.domain, key=lambda d: abs(d - value))
            x_rounded.append([rounded_value])

        return np.row_stack(x_rounded)

    def sample_uniform(self, point_count: int) -> np.ndarray:
        """
        Generates multiple uniformly distributed random parameter points.

        :param point_count: number of data points to generate.
        :returns: Generated points with shape (point_count, num_features)
        """
        indices = np.random.randint(0, len(self.domain), point_count)
        return np.asarray(self.domain)[indices, None]


class InformationSourceParameter(DiscreteParameter):
    def __init__(self, n_sources: int) -> None:
        """
        :param n_sources: Number of information sources in the problem
        """
        super().__init__('source', list(range(n_sources)))
