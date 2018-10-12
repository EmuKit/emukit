# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List

import numpy as np
import GPyOpt

from . import ContinuousParameter
from .discrete_parameter import DiscreteParameter, InformationSourceParameter


class ParameterSpace(object):
    def __init__(self, parameters: List):
        self._parameters = parameters

        # Check no more than one InformationSource parameter
        source_parameter = [param for param in self.parameters if isinstance(param, InformationSourceParameter)]
        if len(source_parameter) > 1:
            raise ValueError('More than one source parameter found')

    @property
    def parameters(self):
        return self._parameters

    @property
    def parameter_names(self):
        return [p.name for p in self._parameters]

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
            else:
                raise NotImplementedError("Nothing except continuous or discrete is supported right now")
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
