# Copyright 2020 Opsani, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Union, Tuple, List, Any, Optional

import numpy as np

from .parameter import Parameter
from .continuous_parameter import ContinuousParameter
from .categorical_parameter import CategoricalParameter
from .discrete_parameter import DiscreteParameter
from .encodings import OneHotEncoding

class BanditParameter(Parameter):
    """
    A multivariate parameter consisting of a restricted domain of the full Cartesian product of its
    constituent sub-parameters
    """
    def __init__(self, name: str, domain: np.ndarray, sub_parameter_names: Optional[List[str]]=None):
        """
        :param name: Name of parameter
        :param domain: List of tuples representing valid values
        :param sub_parameter_names: List of parameters, must correspond to domain if provided,
            otherwise will be reflected from the domain
        """
        super().__init__(name)
        if not isinstance(domain, np.ndarray):
            raise ValueError("Domain must be a 2D np.ndarray, got type: {}".format(type(domain)))
        if not domain.ndim==2:
            raise ValueError("Domain must be a 2D np.ndarray, got dimensions: {}".format(domain.shape))
        self.domain = domain  # each column is homogeneously typed thanks to numpy.ndarray
        self.parameters = self._create_parameters(domain, sub_parameter_names)
        self._sub_parameter_names = sub_parameter_names

    def __str__(self):
        msg = f"<BanditParameter: {self.name} ndim={self.domain.ndim}"
        if self._sub_parameter_names:
            msg = msg + f" ({','.join(self._sub_parameter_names)})"
        msg = msg + '>'
        return msg

    def __repr__(self):
        return f"BanditParameter({self.name}, {self.domain}, {self._sub_parameter_names})"

    def _create_parameter_names(self, domain: np.ndarray) -> List[str]:
        """
        Create names for sub-parameters, used when names are not already provided.

        :param domain: 2D array (n by p) of valid states, each row represents one valid state
        :returns: List of names
        """
        return [f'{self.name}_{i}' for i in range(domain.shape[1])]

    def _create_parameters(self, domain: np.ndarray, parameter_names: Optional[List[str]]) -> List[Parameter]:
        """
        Reflect parameters from domain.

        :param domain: 2D array (n by p) of valid states, each row represents one valid state
        :param parameter_names: Optional list of names for sub-parameters. If not provided,
            sub-parameter names will be automatically generated.
        :returns: List of sub-parameters
        """
        parameters = []
        parameter_names = parameter_names if parameter_names else self._create_parameter_names(domain)
        if domain.shape[1] != len(parameter_names):
            raise ValueError("Provided domain shape {} != number of parameter names {}".format(domain.shape[1], len(parameter_names)))
        for cix, parameter_name in enumerate(parameter_names):
            sub_param_domain = domain[:,cix]
            domain_unq = np.unique(sub_param_domain)
            if np.issubdtype(sub_param_domain.dtype, np.number):  # make discrete
                parameter = DiscreteParameter(name = parameter_name, domain = domain_unq)
            else:  # make categorical
                encoding = OneHotEncoding(domain_unq)
                parameter = CategoricalParameter(name = parameter_name, encoding = encoding)
                raise NotImplementedError("Categorical sub-parameters not yet fully supported")
                # NOTE Categorical sub-parameters not yet implemented because inputs are
                # homogeneously typed np.ndarrays rather than structured arrays. In the future,
                # using structured arrays for all inputs may be more appropriate.
            parameters.append(parameter)
        return parameters

    @property
    def model_parameters(self) -> List:
        return self.parameters

    def check_in_domain(self, x: Union[np.ndarray, float, list]) -> Union[bool, np.ndarray]:
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
            elif x.ndim == 2 and x.shape[1] > 1:
                result = np.array([self.check_in_domain(xx) for xx in x])
                return result
            elif x.ndim > 1:
                raise ValueError("Expected x shape (n,) or (n, 1), actual is {}".format(x.shape))
            if x.shape[0] != self.domain.shape[1]:
                raise ValueError("Received x with dimension {}, expected dimension is {}".format(x.shape[0],self.domain.shape[1]))
            result = (self.domain == x).all(axis=1).any()
        elif isinstance(x, float):
            if self.domain.shape[1] > 1:
                raise ValueError("Received x with dimension 1, expected dimension is {}".format(x.shape[0],self.domain.shape[1]))
            result = (self.domain == x).all(axis=1).any()
        elif isinstance(x, list):
            result = (self.domain == x).all(axis=1).any()
        else:
            raise ValueError("Unsupported type for point x: {}".format(type(x)))
        return result

    @property
    def bounds(self) -> List[Tuple]:
        """
        Calculate the limiting bounds of the sub-parameters

        :returns: a list containing tuples (min, max) for each constituent sub-parameter
        """
        return [pb for p in self.parameters for pb in p.bounds]

    def round(self, x: np.ndarray) -> np.ndarray:
        """
        Rounds each row in x to represent a valid value for this bandit variable. Note that this
        valid value may be 'far' from the suggested value.

        :param x: A 2d array NxD to be rounded (D is len(self.parameters))
        :returns: An array NxD where each row represents a value from the domain
                  that is closest to the corresponding row in x
        """
        if x.ndim != 2:
            raise ValueError("Expected 2d array, got {}".format(x.ndim))

        if x.shape[1] != self.dimension:
            raise ValueError("Expected {} column array, got {}".format(self.dimension, x.shape[1]))

        x_rounded = []
        for row in x:
            dists = np.sqrt(np.sum((self.domain - row)**2))
            rounded_value = min(self.domain, key=lambda d: np.linalg.norm(d-row))
            x_rounded.append(rounded_value)

        if not all([self.check_in_domain(xr) for xr in x_rounded]):
            raise ValueError("Rounding error encountered, not all rounded values in domain.")
        return np.row_stack(x_rounded)

    @property
    def dimension(self) -> int:
        """
        Calculate the aggregate dimensionality of the sub-parameters

        :returns: dimensionality of the BanditParameter
        """
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
        return self.domain[np.random.choice(self.domain.shape[0], point_count)]
