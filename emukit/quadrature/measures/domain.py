# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Tuple

import numpy as np

from ...core.continuous_parameter import ContinuousParameter
from ..typing import BoundsType


class BoxDomain:
    """A box domain defined by a hyper-cube.

    :param name: Name of parameter.
    :param bounds: The bounds defining the box.
                   List of D tuples [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)], where D is
                   the input dimensionality and the tuple (lb_d, ub_d) contains the lower and upper bound
                   of dimension d defining the box.

    """

    def __init__(self, name: str, bounds: BoundsType):
        self._check_bound_validity(bounds)
        self.dim = len(bounds)
        self._bounds = bounds
        self.lower_bounds, self.upper_bounds = self._get_lower_and_upper_bounds()
        self.name = name

    @property
    def bounds(self) -> BoundsType:
        """The bounds defining the hypercube."""
        return self._bounds

    @bounds.setter
    def bounds(self, new_bounds: BoundsType) -> None:
        """Sets new box bounds and checks their validity.

        :param new_bounds: The new bounds.
        """
        if not len(new_bounds) == self.dim:
            raise ValueError("Length of new box bounds is {} (length {} expected).".format(len(new_bounds), self.dim))

        self._check_bound_validity(new_bounds)
        self._bounds = new_bounds
        self.lower_bounds, self.upper_bounds = self._get_lower_and_upper_bounds()

    def _check_bound_validity(self, bounds: BoundsType) -> None:
        """Checks if domain is not empty.

        :param bounds: The bounds to be checked.
        """
        if len(bounds) == 0:
            raise ValueError("Length of bound list must be > 0; empty list found.")

        for bounds_d in bounds:
            lb_d, ub_d = bounds_d
            if lb_d >= ub_d:
                raise ValueError(
                    "Upper box bound must be larger than lower bound. Found a pair containing ({}, "
                    "{}).".format(lb_d, ub_d)
                )

    def _get_lower_and_upper_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Converts the box bounds into two arrays, one containing all lower bounds and one
        containing all upper bounds.

        :return: Lower bounds and upper bounds of box, each of shape (1, input_dim).
        """
        lower_bounds = np.zeros([self.dim, 1])
        upper_bounds = np.zeros([self.dim, 1])
        for i, bounds_d in enumerate(self._bounds):
            lb_d, ub_d = bounds_d
            lower_bounds[i] = lb_d
            upper_bounds[i] = ub_d
        return lower_bounds.T, upper_bounds.T

    def convert_to_list_of_continuous_parameters(self) -> List[ContinuousParameter]:
        """Converts the box bounds into a list of :class:`ContinuousParameter` objects.

        :return: The continuous parameters (one for each dimension of the box).
        """
        continuous_parameters = []
        for i, bounds_d in enumerate(self._bounds):
            lb_d, ub_d = bounds_d
            name_d = self.name + "_" + str(i)
            param = ContinuousParameter(name=name_d, min_value=lb_d, max_value=ub_d)
            continuous_parameters.append(param)
        return continuous_parameters
