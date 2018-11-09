# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Iterable, Union, Tuple

import numpy as np


class DiscreteParameter(object):
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


def create_discrete_parameter_for_categories(name: str, categories: Iterable, encodings: Iterable = None) -> Tuple:
    """
    Creates a DiscreteParameter object for categorical parameter.
    If encodings corresponding to each category value are provided, they are used.
    If encodings are not provided, one-hot encodings are generated.

    :param name: Name of parameter.
    :param categories: List of categories.
    :param encodings: Optional list of corresponding encodings.
    :return: A tuple (DiscreteParameter object, map of categories to encodings, map or encodings to categories)
    """

    if encodings:
        # Encodings are given, perform some validation
        if len(categories) != len(encodings):
            raise ValueError("Number of categories and encodings should match. Actual sizes: {} categories and {} encodings".format(len(categories), len(encodings)))
        if len(set(encodings)) != len(encodings):
            raise ValueError("All encodings should be unique. Found {} duplicates.".format(len(encodings) - len(set(encodings))))
    else:
        # Encodings are not give, generate one-hot values
        encodings = [10 ** i for i, _ in enumerate(categories)]

    param = DiscreteParameter(name, encodings)
    forward_dict = dict(zip(categories, encodings))
    reverse_dict = dict(zip(encodings, categories))

    return param, forward_dict, reverse_dict