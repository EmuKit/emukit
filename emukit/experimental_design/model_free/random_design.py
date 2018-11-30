# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import GPyOpt

from .base import ModelFreeDesignBase


class RandomDesign(ModelFreeDesignBase):
    """
    Random experiment design.
    Random values for all variables within the given bounds.
    """
    def __init__(self, parameter_space):
        super(RandomDesign, self).__init__(parameter_space)
        self.gpyopt_random_design = GPyOpt.experiment_design.RandomDesign(self.gpyopt_design_space)

    def get_samples(self, point_count):
        samples = self.gpyopt_random_design.get_samples(point_count)
        rounded_samples = self.parameter_space.round(samples)
        return rounded_samples
