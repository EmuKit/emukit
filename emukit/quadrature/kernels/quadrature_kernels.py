import numpy as np

from emukit.quadrature.interfaces.standard_kernels import IStandardKernel
from .integral_bounds import IntegralBounds


class QuadratureKernel:
    """
    Abstract class for covariance function of a Gaussian process

    Note that each specific implementation of this class must go with a specific standard kernel as input which
    inherits from IStandardKernel. This is because we both want the QuadratureKernel to be backend agnostic
    but at the same time QuadratureKernel need access to specifics of the standard kernel.

    An example of a specific QuadratureKernel and IStandardKernel pair is QuadratureRBF and IRBF.
    """

    # TODO: change this so it can take alist of tuples rather than the class
    def __init__(self, kern: IStandardKernel, integral_bounds: IntegralBounds) -> None:
        """
        :param kern: standard kernel object IStandardKernel
        :param integral_bounds: define the domain integrated over
        """
        self.kern = kern
        self.bounds = integral_bounds
        self.input_dim = integral_bounds.dim
        self.lower_bounds, self.upper_bounds = integral_bounds.get_bounds_as_separate_arrays()

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


class IDifferentiableKernel:
    """ Adds differentiability to an integrable kernel """

    def dK_dx(self, x:np.ndarray, x2:np.ndarray) -> np.ndarray:
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
