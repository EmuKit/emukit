# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import List, Tuple


class Parameter(object):

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"<Parameter: {self.name}>"

    def __repr__(self):
        return f"Parameter({self.name})"

    @property
    def dimension(self) -> int:
        """
        Gives the dimension of the parameter.
        """
        return 1

    @property
    def model_parameters(self) -> List:
        """
        Gives the list of single dimensional model parameters the parameter corresponds to.
        """
        return [self]

    @property
    def bounds(self) -> List[Tuple]:
        """
        Returns bounds of the parameter in a form of list of tuples
        """
        raise NotImplemented

    def round(self, x: np.ndarray) -> np.ndarray:
        """
        Rounds the values of x to fit to the parameter domain, if needed.

        :param x: 2d array of values to be rounded.
        :returns: A 2d array of rounded values.
        """
        return x

    def check_in_domain(self, x: np.ndarray) -> bool:
        """
        Verifies that given values lie within the parameter's domain

        :param x: Value to be checked
        :return: A boolean value which indicates whether all points lie in the domain
        """
        raise NotImplementedError

    def sample_uniform(self, point_count: int) -> np.ndarray:
        """
        Generates multiple uniformly distributed random parameter points.

        :param point_count: number of data points to generate.
        :returns: Generated points with shape (point_count, num_features)
        """
        raise NotImplementedError
