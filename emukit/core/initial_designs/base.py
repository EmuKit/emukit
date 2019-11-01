# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


class ModelFreeDesignBase(object):
    """
    Base class for all model free experiment designs
    """
    def __init__(self, parameter_space):
        self.parameter_space = parameter_space

    def get_samples(self, point_count):
        raise NotImplementedError("Subclasses should implement this method.")
