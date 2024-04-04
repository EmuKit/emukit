# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List

import numpy as np

from ...core.continuous_parameter import ContinuousParameter
from ..typing import BoundsType


class BoxDomain:
    r"""A box domain defined by a hyper-cube.

    :param bounds: The bounds defining the box.
                   List of :math:`d` tuples :math:`[(a_1, b_1), (a_2, b_2), \dots, (a_d, b_d)]`,
                   where :math:`d` is the dimensionality of the domain and the tuple :math:`(a_i, b_i)`
                   contains the lower and upper bound of dimension :math:`i` defining the box domain.
    :param name: Name of parameter.

    """

    def __init__(self, bounds: BoundsType, name: str = ""):
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

        :raises ValueError: If ``new_bounds`` is not equal to dimensionality of measure.

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

        :raises ValueError: If bounds list is empty.
        :raises ValueError: If volume of hypercube defined by the bounds is emtpy.

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
