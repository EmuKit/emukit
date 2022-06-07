# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import List

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
        self._bounds = bounds
        self.dim = len(bounds)
        self.name = name

        # set upper and lower bounds arrays for convenience
        bounds = np.array(bounds)
        self.lower_bounds = bounds[:, 0]
        self.upper_bounds = bounds[:, 1]

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

        new_bounds = np.array(new_bounds)
        self.lower_bounds = new_bounds[:, 0]
        self.upper_bounds = new_bounds[:, 1]

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
