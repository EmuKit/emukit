# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
try:
    from sobol_seq import i4_sobol_generate
except ImportError:
    raise ImportError('sobol_seq needs to be installed in order to use sobol design')

from .base import ModelFreeDesignBase
from .. import ParameterSpace


class SobolDesign(ModelFreeDesignBase):
    """
    Sobol experiment design.
    Based on sobol_seq implementation. For further reference see https://github.com/naught101/sobol_seq
    """

    def __init__(self, parameter_space: ParameterSpace) -> None:
        """
        param parameter_space: The parameter space to generate design for.
        """
        super(SobolDesign, self).__init__(parameter_space)

    def get_samples(self, point_count: int) -> np.ndarray:
        """
        Generates requested amount of points.

        :param point_count: Number of points required.
        :return: A numpy array of generated samples, shape (point_count x space_dim)
        """
        bounds = self.parameter_space.get_bounds()
        lower_bound = np.asarray(bounds)[:, 0].reshape(1, len(bounds))
        upper_bound = np.asarray(bounds)[:, 1].reshape(1, len(bounds))
        diff = upper_bound - lower_bound

        X_design = np.dot(i4_sobol_generate(len(bounds), point_count), np.diag(diff[0, :])) + lower_bound

        samples = self.parameter_space.round(X_design)

        return samples
