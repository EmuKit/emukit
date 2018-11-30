# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Union

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
