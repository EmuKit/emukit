# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Union, Tuple, List, Any

import numpy as np

from .parameter import Parameter
from .continuous_parameter import ContinuousParameter
from .categorical_parameter import CategoricalParameter
from .discrete_parameter import DiscreteParameter

DomainType = List[Tuple[Any, ...]]

class BanditParameter(Parameter):
    """
    A multivariate parameter consisting of a restricted domain of the full Cartesian product of its
    constituent sub-parameters
    """
    def __init__(self, name: str, domain: DomainType, parameters: Optional[List[Parameters]]=None):
        """
        :param name: Name of parameter
        :param domain: List of tuples representing valid values
        :param parameters: List of parameters, must correspond to domain if provided, otherwise will
        be reflected from the domain
        """
        self.name = name
        self._check_domain_validity(domain)
        self.domain = domain
        if parameters is None:
            self.parameters = self._create_parameters(domain)

    def _check_domain_validity(self, domain: DomainType):
        """ Data validation for the domain. Checks that elements are properly typed i.e. don't mix
        strings and integers. Raises exception if domain is invalid.
        """
        import pdb; pdb.set_trace()

    def _create_parameters(self, domain: DomainType) -> List[Parameters]:
        """ Reflect parameters from domain.
        """
        import pdb; pdb.set_trace()

    def check_in_domain(self, x: Union[np.ndarray, float]) -> bool:
        """
        Checks if all the points in x lie in the domain set

        :param x:    1d numpy array of points to check
                  or 2d numpy array with shape (n_points, 1) of points to check
                  or float of single point to check
        :return: A boolean value which indicates whether all points lie in the domain
        """
        if isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape[1] == 1:
                x = x.ravel()
            elif x.ndim > 1:
                raise ValueError("Expected x shape (n,) or (n, 1), actual is {}".format(x.shape))
        import pdb; pdb.set_trace()
        # return x.totuple() in self.domain
        return np.all([self.min <= x, x <= self.max])

    @property
    def bounds(self) -> List[Tuple]:
        """
        Returns a list containing the bounds for each constituent parameter
        """
        return [p.bounds for p in self.parameters]

    @property
    def dimension(self) -> int:
        d = 0
        for p in self.parameters:
            if isinstance(p, ContinuousParameter): d+=1
            elif isinstance(p, DiscreteParameter): d+=1
            elif isinstance(p, CategoricalParameter): d+=p.dimension
            else: raise Exception("Parameter type {type(p)} not supported.")
        return d

    def sample_uniform(self, point_count: int) -> np.ndarray:
        """
        Generates multiple uniformly distributed random parameter points.

        :param point_count: number of data points to generate.
        :returns: Generated points with shape (point_count, num_features)
        """
        import pdb; pdb.set_trace()
        return np.random.uniform(low=self.min, high=self.max, size=(point_count, 1))
