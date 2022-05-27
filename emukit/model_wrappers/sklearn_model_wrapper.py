from typing import Tuple

import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
except ImportError:
    ImportError("scikit-learn needs to be installed in order to use SklearnGPRWrapper")

from emukit.core.interfaces.models import IModel


class SklearnGPRWrapper(IModel):
    def __init__(self, sklearn_model: GaussianProcessRegressor):
        """
        :param sklearn_model: Scikit-learn GPR model to wrap
        """
        self.model = sklearn_model

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance values for given points

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        mean, std = self.model.predict(X, return_std=True)
        if mean.ndim == 1:
            mean = mean[:, None]
        return mean, np.power(std, 2.0).reshape(-1, 1)

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model
        :param X: new points
        :param Y: function values at new points X

        """
        self.model.X_train_, self.model.y_train_ = X, Y

    def optimize(self) -> None:
        """
        Optimize hyper-parameters of model
        """
        self.model.fit(self.X, self.Y)

    @property
    def X(self) -> np.ndarray:
        return self.model.X_train_

    @property
    def Y(self) -> np.ndarray:
        return self.model.y_train_
