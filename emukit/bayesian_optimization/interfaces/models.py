import numpy as np


class IEntropySearchModel:

    def predict_covariance(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_covariance_between_points(Xi: np.ndarray, Xj: np.ndarray) -> np.ndarray:
        raise NotImplementedError
