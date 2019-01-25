# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List
import numpy as np

from .base import ModelFreeDesignBase


class RandomDesign(ModelFreeDesignBase):
    """
    Random experiment design.
    Random values for all variables within the given bounds.
    """
    def __init__(self, parameter_space):
        super(RandomDesign, self).__init__(parameter_space)

    def get_samples(self, point_count):
        bounds = self.parameter_space.get_bounds()
        X_design = self.samples_multidimensional_uniform(bounds, point_count)
        samples = self.parameter_space.round(X_design)

        return samples

    def samples_multidimensional_uniform(self, bounds: List, point_count: int) -> np.ndarray:
        """
        Generates a multidimensional grid of uniformly distributed random values.

        :param bounds: List of pairs defining the box constraints, in a format (min, max).
        :param point_count: number of data points to generate.
        :returns: Generated grid of random values.
        """
        dim = len(bounds)
        Z_rand = np.zeros(shape=(point_count, dim))
        for k in range(0, dim):
            Z_rand[:, k] = np.random.uniform(low=bounds[k][0], high=bounds[k][1], size=point_count)
        return Z_rand
