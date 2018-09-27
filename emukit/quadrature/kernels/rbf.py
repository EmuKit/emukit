
import numpy as np


class RBF():
    """
    Base class for an RBF kernel
    Inherit from this class to wrap an your custom rbf kernel
    """

    @property
    def lengthscale(self):
        raise NotImplemented

    @property
    def variance(self):
        raise NotImplemented

    def K(self, x, x2):
        """
        The kernel evaluated at x and x2

        :param x: the first argument of the kernel with shape (number of points N, input_dim)
        :param x2: the second argument of the kernel with shape (number of points M, input_dim)

        :return: the kernel matrix with shape (N, M)
        """
        raise NotImplemented

    def dK_dx(self, x: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Gradient of the kernel (might be required if an integrable kernel with gradients is constructed)

        :param x: first location at which to evaluate the kernel and component wrt which derivative has been taken
        :param x2: second location to evaluate

        :return: the gradient at x
        """
        raise NotImplemented