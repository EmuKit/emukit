import numpy as np


class UserFunctionResult(object):
    """
    A class that records the inputs, outputs and meta-data of an evaluation of the user function.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        :param X: Function input, 1 by function input dimension
        :param Y: Function output(s), 1 by function output dimension
        """
        if X.ndim != 1:
            raise ValueError("x is expected to be 1-dimensional, actual dimentionality is {}".format(X.ndim))

        if Y.ndim != 1:
            raise ValueError("y is expected to be 1-dimensional, actual dimentionality is {}".format(Y.ndim))

        self.X = X
        self.Y = Y
