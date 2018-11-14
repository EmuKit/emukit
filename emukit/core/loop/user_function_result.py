# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np


class UserFunctionResult(object):
    """
    A class that records the inputs, outputs and meta-data of an evaluation of the user function.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, cost: np.ndarray=None) -> None:
        """
        :param X: Function input, 1 by function input dimension
        :param Y: Function output(s), 1 by function output dimension
        :param cost: Cost of evaluating the function, 1 by function cost dimension
        """
        if X.ndim != 1:
            raise ValueError("x is expected to be 1-dimensional, actual dimensionality is {}".format(X.ndim))

        if Y.ndim != 1:
            raise ValueError("y is expected to be 1-dimensional, actual dimensionality is {}".format(Y.ndim))

        if cost is not None and cost.ndim != 1:
            raise ValueError("cost is expected to be 1-dimensional, actual dimensionality is {}".format(cost.ndim))

        self.X = X
        self.Y = Y
        self.cost = cost
