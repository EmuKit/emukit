# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import itertools
from typing import List, Optional, Tuple

import numpy as np

from .constraints import IConstraint
from .discrete_parameter import InformationSourceParameter
from .parameter import Parameter


class ParameterSpace(object):
    """
    Represents parameter space for a given problem.
    """

    def __init__(self, parameters: List, constraints: Optional[List[IConstraint]]=None):
        """
        Creates a new instance of a parameter space.

        :param parameters: A list of parameters in the space.
        :param constraints: A list of constraints on the input domain
        """
        self._parameters = parameters

        if constraints:
            self.constraints = constraints
        else:
            self.constraints = []

        # Check no more than one InformationSource parameter
        source_parameter = [param for param in self.parameters if isinstance(param, InformationSourceParameter)]
        if len(source_parameter) > 1:
            raise ValueError('More than one source parameter found')

        # Check uniqueness of parameter names
        names = self.parameter_names
        if not len(names) == len(set(names)):
            raise ValueError('Parameter names are not unique')

    def find_parameter_index_in_model(self, parameter_name: str) -> List[int]:
        """
        Find the indices of the encoding of the specified parameter in the input vector

        :param parameter_name: Parameter name to find indices for
        :return: List of indices
        """
        i_start = 0
        for param in self._parameters:
            if param.name == parameter_name:
                return list(range(i_start, i_start + param.dimension))
            else:
                i_start += param.dimension
        raise ValueError('Parameter {} not found'.format(parameter_name))

    @property
    def dimensionality(self):
        return sum([p.dimension for p in self._parameters])

    @property
    def parameters(self) -> List:
        """
        Returns the list of parameters in the space.
        """

        return self._parameters

    @property
    def parameter_names(self) -> List:
        """
        Returns the list of names of parameters in the space.
        """

        return [p.name for p in self._parameters]

    def get_parameter_by_name(self, name: str) -> Parameter:
        """
        Returns parameter with the given name

        :param name: Parameter name
        :returns: A parameter object
        """

        for param in self.parameters:
            if param.name == name:
                return param
        raise ValueError('Parameter with name ' + name + ' not found.')

    def get_bounds(self) -> List[Tuple]:
        """
        Returns a list of tuples containing the min and max value each parameter can take.

        If the parameter space contains categorical variables, the min and max values correspond to each variable used
        to encode the categorical variables.
        """

        # bounds is a list of lists
        bounds = [param.bounds for param in self.parameters]
        # Convert list of lists to one list
        return list(itertools.chain.from_iterable(bounds))

    def round(self, x: np.ndarray) -> np.ndarray:
        """
        Rounds given values x to closest valid values within the space.

        :param x: A 2d array of values to be rounded
        :returns: A 2d array of rounded values
        """

        x_rounded = []
        current_idx = 0
        for param in self.parameters:
            param_columns = x[:, current_idx:(current_idx + param.dimension)]
            x_rounded.append(param.round(param_columns))
            current_idx += param.dimension

        return np.column_stack(x_rounded)

    def check_points_in_domain(self, x: np.ndarray) -> np.ndarray:
        """
        Checks that each column of x lies in the domain of the corresponding parameter

        :param x: 2d numpy array of points to check
        :return: A 1d numpy array which contains a boolean indicating whether each point is in domain
        """
        len_encoding = sum(len(param.model_parameters) for param in self.parameters)
        if x.shape[1] != len_encoding:
            raise ValueError('x should have number of columns equal to the sum'
                             'of all parameter encodings, expected {} actual {}'
                             .format(x.shape[1], len_encoding))

        in_domain = np.ones(x.shape[0], dtype=bool)
        encoding_index = 0
        for param in self._parameters:
            # First check if this particular parameter is in domain
            param_in_domain = [
                param.check_in_domain(x[[point_ix], encoding_index:(encoding_index + param.dimension)])
                for point_ix in range(x.shape[0])]
            # Set in_domain to be False if this parameter or any previous parameter is out of domain
            in_domain = np.all([in_domain, param_in_domain], axis=0)
            encoding_index += param.dimension
        return in_domain

    def sample_uniform(self, point_count: int) -> np.ndarray:
        """
        Generates multiple uniformly distributed random parameter points.

        :param point_count: number of data points to generate.
        :returns: Generated points with shape (point_count, num_features)
        """
        parameter_samples = [param.sample_uniform(point_count) for param in self.parameters]
        return np.hstack(parameter_samples)
