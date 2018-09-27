import numpy as np

from from emukit.core import IntegralBounds


class IntegrableKernel():
    """ Abstract class for covariance function of a Gaussian process """

    def __init__(self, input_dim: int, integral_bounds: IntegralBounds) -> None:
        """
        :param input_dim: Input dimension
        :param integral_bounds: define the domain integrated over
        """
        self.input_dim = input_dim
        self.lower_bounds, self.upper_bounds = integral_bounds.get_bounds()

    def K(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        definition of covariance function (aka. kernel) k(x,x')

        :param x1: first argument of the kernel
        :param x2: second argument of the kernel

        :returns: kernel evaluated at x1, x2
        """
        raise NotImplementedError

    def qK(self, x2: np.ndarray) -> np.ndarray:
        """
        Kernel with the first component integrated out aka. kernel mean

        :param x2: only remaining argument of the once integrated kernel

        :returns: of kernel mean at locations x2
        """
        raise NotImplementedError

    def Kq(self, x1: np.ndarray) -> np.ndarray:
        """
        Kernel with the second component integrated out aka. kernel mean

        :param x1: only remaining argument of the once integrated kernel

        :returns: kernel mean at locations x1
        """
        raise NotImplementedError

    def qKq(self) -> np.float:
        """
        Kernel integrated over both arguments x1 and x2

        :returns: double integrated kernel
        """
        raise NotImplementedError


class IDifferentiableKernel():
    """ Adds differentiability to an integrable kernel """

    def dK_dx(self) -> np.ndarray:
        """
        gradient of the kernel wrt x

        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)
        :param x2: M points at which to evaluate, np.ndarray with x2.shape = (M, input_dim)

        :return: the gradient of K with shape (input_dim, N, M)
        """
        raise NotImplementedError

    def dqK_dx(self, x: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel mean evaluated at x
        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)

        :return: the gradient with shape (input_dim, N)
        """
        raise NotImplementedError

    def dKq_dx(self, x:np.ndarray) -> np.ndarray:
        """
        gradient of the transposed kernel mean evaluated at x
        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)

        :return: the gradient with shape (N, input_dim)
        """
        raise NotImplementedError


class IntegrableKernelWithGradients(IntegrableKernel, IDifferentiableKernel):
    """
    An integrable kernel that is also differentiable
    """
    def __init__(self, input_dim: int, integral_bounds: IntegralBounds) -> None:
        super(IntegrableKernelWithGradients, self).__init__(input_dim, integral_bounds)
