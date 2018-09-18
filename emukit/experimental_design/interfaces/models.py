import numpy as np


class ICalculateVarianceReduction:
    def calculate_variance_reduction(self, x_train_new: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        """
        Computes the variance reduction at x_test, if a new point at x_train_new is acquired
        """
        raise NotImplementedError