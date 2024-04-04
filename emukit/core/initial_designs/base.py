# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from .. import ParameterSpace


class InitialDesignBase(object):
    """
    Base class for all initial designs
    """

    def __init__(self, parameter_space: ParameterSpace):
        """
        :param parameter_space: The parameter space to generate design for.

        """
        self.parameter_space = parameter_space

    def get_samples(self, point_count: int) -> np.ndarray:
        """
        Generates requested amount of points.

        :param point_count: Number of points required.
        :return: A numpy array of generated samples, shape (point_count x space_dim)
        """
        raise NotImplementedError("Subclasses should implement this method.")
