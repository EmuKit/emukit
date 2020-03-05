# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
try:
    import pyDOE
except ImportError:
    raise ImportError('pyDOE needs to be installed in order to use latin design')

from .base import ModelFreeDesignBase
from .. import ParameterSpace


class LatinDesign(ModelFreeDesignBase):
    """
    Latin hypercube experiment design.

    Based on pyDOE implementation. For further reference see https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube
    """
    def __init__(self, parameter_space: ParameterSpace) -> None:
        """
        :param parameter_space: The parameter space to generate design for.
        """
        super(LatinDesign, self).__init__(parameter_space)

    def get_samples(self, point_count: int) -> np.ndarray:
        """
        Generates requested amount of points.

        :param point_count: Number of points required.
        :return: A numpy array of generated samples, shape (point_count x space_dim)
        """
        bounds = self.parameter_space.get_bounds()
        X_design_aux = pyDOE.lhs(len(bounds), point_count, criterion='center')
        ones = np.ones((X_design_aux.shape[0], 1))

        lower_bound = np.asarray(bounds)[:, 0].reshape(1, len(bounds))
        upper_bound = np.asarray(bounds)[:, 1].reshape(1, len(bounds))
        diff = upper_bound - lower_bound

        X_design = np.dot(ones, lower_bound) + X_design_aux * np.dot(ones, diff)

        samples = self.parameter_space.round(X_design)

        return samples
