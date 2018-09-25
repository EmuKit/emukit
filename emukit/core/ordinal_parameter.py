from typing import Iterable

import numpy as np


class OrdinalParameter(object):
    """
    A parameter that takes a discrete set of values where the order is important
    """
    def __init__(self, name: str, domain: Iterable):
        """

        :param name: Name of parameter
        :param domain: valid values the parameter can have
        """
        self.name = name
        self.domain = domain

    def check_in_domain(self, x: Iterable) -> bool:
        """
        Checks if the points in x are in the set of allowed values

        :param x: 1d numpy array of points to check
        :return: A 1d numpy array which contains a boolean indicating whether each point is in domain
        """
        return set(x).issubset(set(self.domain))


class InformationSourceParameter(OrdinalParameter):
    def __init__(self, n_sources: int) -> None:
        """
        :param n_sources: Number of information sources in the problem
        """
        super().__init__('source', list(range(n_sources)))
