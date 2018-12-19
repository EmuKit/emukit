# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import List

from .encodings import Encoding
from .parameter import Parameter
from .continuous_parameter import ContinuousParameter


class CategoricalParameter(Parameter):
    def __init__(self, name: str, encoding: Encoding):
        self.name = name

        # ensure float just in case we were given integers
        self.encoding = encoding

        self._cont_params = []
        for column_idx in range(self.encodings.shape[1]):
            cont_param = ContinuousParameter(name + '_' + str(column_idx),
                                             np.min(self.encodings[:, column_idx]),
                                             np.max(self.encodings[:, column_idx]))
            self._cont_params.append(cont_param)

    @property
    def encodings(self) -> np.ndarray:
        return self.encoding.encodings

    @property
    def model_parameters(self) -> List:
        return self._cont_params

    def round(self, x: np.ndarray) -> np.ndarray:
        return self.encoding.round(x)

    @property
    def dimension(self) -> int:
        return self.encodings.shape[1]

    def check_in_domain(self, x: np.ndarray) -> bool:
        """
        Verifies that given values lie within the parameter's domain

        :param x: Value to be checked
        :return: A boolean value which indicates whether all points lie in the domain
        """
        for i, param in enumerate(self._cont_params):
            # First check if this particular parameter is in domain
            param_in_domain = param.check_in_domain(x[:, i])
            if not param_in_domain:
                return False

        return True
