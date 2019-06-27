# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np


class UserFunctionResult(object):
    """
    A class that records the inputs, outputs and meta-data of an evaluation of the user function.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> None:
        """
        :param X: Function input, 1 by function input dimension
        :param Y: Function output(s), 1 by function output dimension
        :param cost: Cost of evaluating the function, 1 by function cost dimension
        """
        if X.ndim != 1:
            raise ValueError("x is expected to be 1-dimensional, actual dimensionality is {}".format(X.ndim))

        if Y.ndim != 1:
            raise ValueError("y is expected to be 1-dimensional, actual dimensionality is {}".format(Y.ndim))

        self.extra_outputs = dict()
        for (key, val) in kwargs.items():
            if val.ndim != 1:
                raise ValueError('Key word arguments must be 1-dimensional but {} is {}d'.format(key, val.ndim))
            self.extra_outputs[key] = val

        self.X = X
        self.Y = Y

    def __getattr__(self, item):
        """
        Allow extra output values to be accessed as an attribute

        :param item: The name of the extra output to be accessed
        :return: The value of the extra output
        """
        return self.extra_outputs[item]
