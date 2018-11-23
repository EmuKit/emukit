# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List

import numpy as np
import GPyOpt

from . import Parameter
from . import ContinuousParameter
from .discrete_parameter import DiscreteParameter, InformationSourceParameter






class CategoricalParameter(Parameter):
    def __init__(self, name: str, categories: List, encodings: np.ndarray):
        self.name = name

        self.categories = categories
        self.encodings = encodings

    def transform_to_model(self, category):
        idx = categories.index(category)
        return encodings[idx]

    def transform_to_user_function(self, x):
        max_idx = np.argmax(x)
        new_x = np.zeros(x.shape, dtype)
        new_x[max_idx] = 1

    @property
    def model_dim(self):
        return self.encodings.shape[1]

    # def check_in_domain(self, x: Union[np.ndarray, str]) -> bool:
    #     """
        

    #     :param x: 
    #     :return: A boolean value which indicates whether all points lie in the domain
    #     """
    #     return np.all([(self.min < x), (self.max > x)], axis=0)

class ParameterSpace(object):
    def __init__(self, parameters: List):
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
    def parameters(self):
        return self._parameters

    @property
    def parameter_names(self):
        return [p.name for p in self._parameters]

    def get_parameter_by_name(self, name):
        for param in self.parameters:
            if param.name == name:
                return param
        raise ValueError('Parameter with name ' + name + ' not found.')

    def transform_to_model(self, x):
        result = []
        for i, param in enumerate(self.parameters):
            param_column = x[:, i:(i + 1)]
            result.append(param.transform_to_model(param_column))

        return np.column_stack(result)

    def transform_to_user_function(self, x):
        result = []
        current_idx = 0
        for param in self.parameters:
            param_columns = x[:, current_idx:(current_idx + param.model_dim)]
            result.append(param.transform_to_user_function(param_columns))
            current_idx += param.model_dim

        return np.column_stack(result)

    def convert_to_gpyopt_design_space(self):
        """
        Converts this ParameterSpace to a GPyOpt DesignSpace object
        """

        gpyopt_parameters = []

        for parameter in self.parameters:
            if isinstance(parameter, ContinuousParameter):
                gpyopt_param = {'name': parameter.name, 'type': 'continuous', 'domain': (parameter.min, parameter.max),
                                'dimensionality': 1}
            elif isinstance(parameter, DiscreteParameter):
                gpyopt_param = {'name': parameter.name, 'type': 'discrete', 'domain': parameter.domain,
                                'dimensionality': 1}
            elif isinstance(parameter, CategoricalParameter):
                gpyopt_param = {'name': parameter.name, 'type': 'categorical', 'domain': [i for i, _ in enumerate(parameter.categories)]}
            else:
                raise NotImplementedError("Only continuous, discrete and categorical parameters are supported, received " + type(parameter))
            gpyopt_parameters.append(gpyopt_param)

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
