# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, List

from ...core.continuous_parameter import ContinuousParameter


class BoxBounds:
    """
    Box bounds
    """
    def __init__(self, name: str, bounds: List[Tuple[float, float]]):
        """
        :param name: Name of parameter
        :param bounds: List of D tuples, where D is the dimensionality and the tuples contain the
        lower and upper bounds of the box i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        """

        self.name = name
        self.dim = len(bounds)
        self._check_bound_validity(bounds)
        self._bounds = bounds
        self.lower_bounds, self.upper_bounds = self.get_lower_and_upper_bounds()

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, new_bounds: List[Tuple[float, float]]) -> None:
        """
        Sets new box bounds and checks their validity
        :param new_bounds: List of D tuples, where D is the dimensionality of the box and the tuples contain the
        lower and upper bounds of the box i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        """
        if not len(new_bounds) == self.dim:
            raise ValueError('Length of new box bounds is {} (length {} expected).'.format(len(new_bounds), self.dim))

        self._check_bound_validity(new_bounds)
        self._bounds = new_bounds
        self.lower_bounds, self.upper_bounds = self.get_lower_and_upper_bounds()

    def _check_bound_validity(self, bounds: List[Tuple[float, float]]) -> None:
        """
        checks if lower bounds are smaller than upper bounds.
        :param bounds: List of D tuples, where D is the dimensionality of the box and the tuples contain the
        lower and upper bounds of the box i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        """
        if self.dim == 0:
            raise ValueError("Length of bound list must be > 0; empty list found.")
        for bounds_d in bounds:
            lb_d, ub_d = bounds_d
            if lb_d >= ub_d:
                raise ValueError("Upper box bound must be larger than lower bound. Found a pair containing ({}, "
                                 "{}).".format(lb_d, ub_d))

    def get_lower_and_upper_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns two arrays, one containing all lower bounds and one containing all upper bounds.
        :return: lower bounds, upper bounds of box, shapes (1, input_dim)
        """
        lower_bounds = np.zeros([self.dim, 1])
        upper_bounds = np.zeros([self.dim, 1])
        for i, bounds_d in enumerate(self._bounds):
            lb_d, ub_d = bounds_d
            lower_bounds[i] = lb_d
            upper_bounds[i] = ub_d
        return lower_bounds.T, upper_bounds.T

    def convert_to_list_of_continuous_parameters(self) -> List[ContinuousParameter]:
        """
        converts the box bounds into a list of ContinuousParameter objects
        :return: a list if ContinuousParameter objects (one for each dimension)
        """
        continuous_parameters = []
        for i, bounds_d in enumerate(self._bounds):
            lb_d, ub_d = bounds_d
            name_d = self.name + '_' + str(i)
            param = ContinuousParameter(name=name_d, min_value=lb_d, max_value=ub_d)
            continuous_parameters.append(param)
        return continuous_parameters
