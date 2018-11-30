# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Iterable, Union

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
        self.name = name
        self.domain = domain

    def check_in_domain(self, x: Union[Iterable, float]) -> bool:
        """
        Checks if the points in x are in the set of allowed values

        :param x: 1d numpy array of points to check
        :return: A boolean indicating whether each point is in domain
        """
        if np.isscalar(x):
            x = [x]
        return set(x).issubset(set(self.domain))


class InformationSourceParameter(DiscreteParameter):
    def __init__(self, n_sources: int) -> None:
        """
        :param n_sources: Number of information sources in the problem
        """
        super().__init__('source', list(range(n_sources)))
