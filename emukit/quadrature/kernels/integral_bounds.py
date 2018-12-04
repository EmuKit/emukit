# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from typing import Tuple, List

from emukit.core.continuous_parameter import ContinuousParameter


class IntegralBounds:
    """
    The integral bounds i.e., the edges of the domain of the integral
    """
    def __init__(self, name: str, bounds: List):
        """
        :param name: Name of parameter
        :param bounds: List of D tuples, where D is the dimensionality of the integral and the tuples contain the
        lower and upper bounds of the integral i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        """

        self.name = name
        self.bounds = bounds
        self.dim = len(bounds)

        self._check_bound_validity()

    def _check_bound_validity(self) -> None:
        """
        checks if lower bounds are smaller than upper bounds.
        """
        for bounds_d in self.bounds:
            lb_d, ub_d = bounds_d
            if lb_d >= ub_d:
                raise ValueError("Upper integral bound must be larger than lower bound")

    def get_lower_and_upper_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns two arrays, one containing all lower bounds and one containing all upper bounds.
        :return: lower bounds, upper bounds of integral, shapes (1, input_dim)
        """
        lower_bounds = np.zeros([self.dim, 1])
        upper_bounds = np.zeros([self.dim, 1])
        for i, bounds_d in enumerate(self.bounds):
            lb_d, ub_d = bounds_d
            lower_bounds[i] = lb_d
            upper_bounds[i] = ub_d
        return lower_bounds.T, upper_bounds.T

    def convert_to_list_of_continuous_parameters(self) -> List[ContinuousParameter]:
        """
        converts the integral bounds into a list of ContinuousParameter objects
        :return: a list if ContinuousParameter objects (one for each dimension)
        """
        continuous_parameters = []
        for i, bounds_d in enumerate(self.bounds):
            lb_d, ub_d = bounds_d
            name_d = self.name + '_' + str(i)
            param = ContinuousParameter(name=name_d, min_value=lb_d, max_value=ub_d)
            continuous_parameters.append(param)
        return continuous_parameters
