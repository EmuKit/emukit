# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import List, Tuple

from .encodings import Encoding
from .parameter import Parameter
from .continuous_parameter import ContinuousParameter


class CategoricalParameter(Parameter):
    def __init__(self, name: str, encoding: Encoding):
        super().__init__(name)

        # ensure float just in case we were given integers
        self.encoding = encoding

        self._cont_params = []
        for column_idx in range(self.encodings.shape[1]):
            cont_param = ContinuousParameter(name + '_' + str(column_idx),
                                             np.min(self.encodings[:, column_idx]),
                                             np.max(self.encodings[:, column_idx]))
            self._cont_params.append(cont_param)

    def __str__(self):
        return f"<CategoricalParameter: {self.name} n_cat={self.dimension}>"

    def __repr__(self):
        return f"CategoricalParameter({self.name}, {self.encoding})"

    @property
    def encodings(self) -> np.ndarray:
        return self.encoding.encodings

    @property
    def model_parameters(self) -> List:
        return self._cont_params

    def round(self, x: np.ndarray) -> np.ndarray:
        return self.encoding.round(x)

    @property
    def bounds(self) -> List[Tuple]:
        """
        Returns a list of tuples containing where each tuple contains the minimum and maximum of the variables used to
        encode the categorical parameter..
        """
        return [(param.min, param.max) for param in self._cont_params]

    @property
    def dimension(self) -> int:
        return self.encodings.shape[1]

    def check_in_domain(self, x: np.ndarray) -> bool:
        """
        Verifies that given values lie within the parameter's domain

        :param x: 2d numpy array with shape (points, encoding) of points to check
        :return: A boolean value which indicates whether all points lie in the domain
        """
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError("Expected x shape (points, {}), actual is {}"
                             .format(self.dimension, x.shape))

        for i, param in enumerate(self._cont_params):
            # First check if this particular parameter is in domain
            param_in_domain = param.check_in_domain(x[:, i])
            if not param_in_domain:
                return False

        return True

    def sample_uniform(self, point_count: int) -> np.ndarray:
        """
        Generates multiple uniformly distributed random parameter points.

        :param point_count: number of data points to generate.
        :returns: Generated points with shape (point_count, num_features)
        """
        indices = np.random.randint(0, self.encodings.shape[0], point_count)
        return self.encodings[indices]
