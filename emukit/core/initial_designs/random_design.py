# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from .base import ModelFreeDesignBase
from .. import ParameterSpace


class RandomDesign(ModelFreeDesignBase):
    """
    Random experiment design.
    Uniform random values for all variables within the given bounds.
    """
    def __init__(self, parameter_space: ParameterSpace) -> None:
        """
        :param parameter_space: The parameter space to generate design for.
        """
        super(RandomDesign, self).__init__(parameter_space)

    def get_samples(self, point_count: int) -> np.ndarray:
        """
        Generates requested amount of points.

        :param point_count: Number of points required.
        :return: A numpy array of generated samples, shape (point_count x space_dim)
        """
        return self.parameter_space.sample_uniform(point_count)
