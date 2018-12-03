# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import GPyOpt

from .base import ModelFreeDesignBase


class LatinDesign(ModelFreeDesignBase):
    """
    Latin hypercube experiment design.
    """
    def __init__(self, parameter_space):
        super(LatinDesign, self).__init__(parameter_space)
        self.gpyopt_latin_design = GPyOpt.experiment_design.LatinDesign(self.gpyopt_design_space)

    def get_samples(self, point_count):
        samples = self.gpyopt_latin_design.get_samples(point_count)
        rounded_samples = self.parameter_space.round(samples)
        return rounded_samples
