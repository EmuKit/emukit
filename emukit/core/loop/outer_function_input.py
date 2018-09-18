import numpy as np


class OuterFunctionInput(object):
    """
    This defines the inputs to the function
    """
    def __init__(self, X: np.ndarray) -> None:
        """
        :param X: Input location
        """
        self.X = X
