
import GPy
from .rbf import RBF


class RBFGPy(RBF):
    """
    Wrapper around the GPy RBF kernel
    """

    def __init__(self, input_dim, variance=1., lengthscale=1.):
        """
        :param input_dim: Input dimension
        :param variance: Squared output scale
        :param lengthscale: Lengthscale of the kernel
        """
        super(RBFGPy, self).__init__(input_dim=input_dim)

        self.gpy_rbf = GPy.kern.src.rbf.RBF(input_dim=input_dim, variance=variance, lengthscale=lengthscale)

    @property
    def lengthscale(self):
        return self.gpy_rbf.lengthscale

    @property
    def variance(self):
        return self.gpy_rbf.variance


    def K(self, x, x2=None):
        """
        The kernel evaluated at x and x2

        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)
        :param x2: M points at which to evaluate, np.ndarray with x2.shape = (M, input_dim)

        :return: the gradient of K with shape (N, M)
        """
        return self.gpy_rbf.K(x, x2)

    def dK_dx(self, x, x2):
        """
        gradient of the kernel wrt x

        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)
        :param x2: M points at which to evaluate, np.ndarray with x2.shape = (M, input_dim)

        :return: the gradient of K with shape (input_dim, N, M)
        """
        return self.gpy_rbf.dK_dr_via_X(x, x2)[None, ...] * self._dr_dx(x, x2)

    # helper
    def _dr_dx(self, x, x2):
        """
        Derivative of the radius

        .. math::

            r = \sqrt{ \frac{||x - x_2||^2}{\lambda^2} }

        name mapping:
            \lambda: self.rbf.lengthscale

        :param x: N points at which to evaluate, np.ndarray with x.shape = (N, input_dim)
        :param x2: M points at which to evaluate, np.ndarray with x2.shape = (M, input_dim)

        :return: the gradient of K with shape (input_dim, N, M)
        """
        return (x.T[:, :, None] - x2.T[:, None, :]) / \
               (self.lengthscale ** 2 * (x.T[:, :, None] - x2.T[:, None, :]) / (self.lengthscale * np.sqrt(2)))
