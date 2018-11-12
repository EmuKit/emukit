# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Union

import numpy as np
from itertools import count


class ContinuousParameter(object):
    """
    A univariate continuous parameter with a domain defined in a range between two values
    """
    def __init__(self, name: str, min_value: float, max_value: float):
        """
        :param name: Name of parameter
        :param min_value: Minimum value the parameter is allowed to take
        :param max_value: Maximum value the parameter is allowed to take
        """
        self.name = name
        self.min = min_value
        self.max = max_value

    def check_in_domain(self, x: Union[np.ndarray, float]) -> bool:
        """
        Checks if all the points in x lie between the min and max allowed values

        :param x: 1d numpy array of points to check or float of single point to check
        :return: A boolean value which indicates whether all points lie in the domain
        """
        return np.all([(self.min < x), (self.max > x)], axis=0)


class MultiDimensionalContinuousParameter(object):
    """
    A multivariate continuous parameter with a domain defined in a range between two values per dimension
    """
    def __init__(self, name: str, min_values: np.ndarray, max_values: np.ndarray):
        """
        :param name: Name of parameter
        :param min_value: Minimum value the parameter is allowed to take (lower bounds of domain), shape (1, dimension)
        :param max_value: Maximum value the parameter is allowed to take (upper bounds of domain), shape (1, dimension)
        """
        self.name = name
        self.lower_bounds = min_values
        self.upper_bounds = max_values

    def check_in_domain(self, x: np.ndarray) -> np.ndarray:
        """
        Checks if the points in x lie between the min and max allowed values
        :param x: locations (n_points, input_dim)
        :return: a boolean array (n_points,) indicating whether each point is in domain
        """
        return np.all([np.all(self.lower_bounds < x, axis=1), np.all(self.upper_bounds > x, axis=1)], axis=0)

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds

    def convert_to_list_of_continuous_parameters(self):
        """
        This iterates through the dimensions and creates one ContinuousParameter object for each pair of lower_bound
        and upper_bound.

        :return: list if ContinuousParameter objects
        """
        lower_bounds, upper_bounds = self.get_bounds()
        continuous_parameters = []
        for i, lb, ub in zip(count(), lower_bounds, upper_bounds):
            name_i = self.name+'_'+str(i)
            cont_param = ContinuousParameter(name=name_i, min_value=lower_bounds[i], max_value=upper_bounds[i])
            continuous_parameters.append(cont_param)
        return continuous_parameters
