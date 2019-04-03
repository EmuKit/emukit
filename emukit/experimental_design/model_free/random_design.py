# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .base import ModelFreeDesignBase


class RandomDesign(ModelFreeDesignBase):
    """
    Random experiment design.
    Random values for all variables within the given bounds.
    """
    def __init__(self, parameter_space):
        super(RandomDesign, self).__init__(parameter_space)

    def get_samples(self, point_count):
        return self.parameter_space.sample_uniform(point_count)
