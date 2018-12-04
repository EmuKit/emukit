# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List

import numpy as np
import GPyOpt

from .parameter import Parameter
from . import CategoricalParameter
from . import ContinuousParameter
from .discrete_parameter import DiscreteParameter, InformationSourceParameter


class ParameterSpace(object):
    """
    Represents parameter space for a given problem.
    """

    def __init__(self, parameters: List):
        """
        Creates a new instance of a parameter space.

        :param parameters: A list of parameters in the space.
        """
        self._parameters = parameters

        # Check no more than one InformationSource parameter
        source_parameter = [param for param in self.parameters if isinstance(param, InformationSourceParameter)]
        if len(source_parameter) > 1:
            raise ValueError('More than one source parameter found')

        # Check uniqueness of parameter names
        names = self.parameter_names
        if not len(names) == len(set(names)):
            raise ValueError('Parameter names are not unique')

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

    def convert_to_gpyopt_design_space(self):
        """
        Converts this ParameterSpace to a GPyOpt DesignSpace object
        """

        gpyopt_parameters = []

        for parameter in self.parameters:
            if isinstance(parameter, ContinuousParameter):
                gpyopt_param = {'name': parameter.name, 'type': 'continuous', 'domain': (parameter.min, parameter.max),
                                'dimensionality': 1}
                gpyopt_parameters.append(gpyopt_param)
            elif isinstance(parameter, DiscreteParameter):
                gpyopt_param = {'name': parameter.name, 'type': 'discrete', 'domain': parameter.domain,
                                'dimensionality': 1}
                gpyopt_parameters.append(gpyopt_param)
            elif isinstance(parameter, CategoricalParameter):
                for i, cat_sub_param in enumerate(parameter.model_parameters):
                    gpyopt_param = {'name': parameter.name + '_' + str(i),
                                    'type': 'continuous',
                                    'domain': (cat_sub_param.min, cat_sub_param.max),
                                    'dimensionality': 1}
                    gpyopt_parameters.append(gpyopt_param)
            else:
                raise NotImplementedError("Only continuous, discrete and categorical parameters are supported"
                                          ", received " + type(parameter))

        return GPyOpt.core.task.space.Design_space(gpyopt_parameters)

    def check_points_in_domain(self, x: np.ndarray) -> np.ndarray:
        """
        Checks that each column of x lies in the domain of the corresponding parameter

        :param x: 2d numpy array of points to check
        :return: A 1d numpy array which contains a boolean indicating whether each point is in domain
        """
        if x.shape[1] != len(self.parameters):
            raise ValueError('x should have number of columns equal to dimensionality of the parameter space')

        in_domain = np.ones(x.shape[0], dtype=bool)
        for i, param in enumerate(self._parameters):
            # First check if this particular parameter is in domain
            param_in_domain = param.check_in_domain(x[:, i])
            # Set in_domain to be False if this parameter or any previous parameter is out of domain
            in_domain = np.all([in_domain, param_in_domain], axis=0)
        return in_domain
