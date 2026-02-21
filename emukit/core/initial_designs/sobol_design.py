# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.stats import qmc

from .. import ParameterSpace
from .base import InitialDesignBase


class SobolDesign(InitialDesignBase):
    """
    Sobol experiment design.
    Based on scipy implementation. For further reference see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html
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
        d = len(bounds)
        lower_bounds = [x[0] for x in bounds]
        upper_bounds = [x[1] for x in bounds]

        sampler = qmc.Sobol(d)
        samples = sampler.random(n=point_count)
        samples = qmc.scale(samples, lower_bounds, upper_bounds)

        X_design = self.parameter_space.round(samples)

        return X_design
