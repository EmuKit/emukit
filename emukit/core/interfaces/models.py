import numpy as np
from typing import Tuple


class IModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for given points

        :param X: points to run prediction for
        """
        raise NotImplementedError

    def update_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Updates model with new data points.

        :param X: new points
        :param Y: function values at new points X
        """
        raise NotImplementedError

    def optimize(self) -> None:
        """
        Optimize hyper-parameters of model
        """
        raise NotImplementedError

    @property
    def X(self):
        raise NotImplementedError

    @property
    def Y(self):
        raise NotImplementedError


class IDifferentiable:
    def get_prediction_gradients(self, X: np.ndarray) -> Tuple:
        """
        Computes and returns model gradients of mean and variance at given points

        :param X: points to compute gradients at
        """
        raise NotImplementedError
