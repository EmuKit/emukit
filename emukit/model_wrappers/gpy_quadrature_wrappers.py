
from emukit.quadrature.interfaces.rbf import IRBF

from emukit.quadrature.kernels import IntegrableKernel, IntegrableRBF
from emukit.quadrature.interfaces import IBaseGaussianProcess


class GPRegressionGPy(IBaseGaussianProcess):
    """
    Wrapper for GPy GPRegression

    An instance of this can be passed as 'base_gp' to an ApproximateWarpedGPSurrogate object
    """
    def __init__(self, gpy_model, kernel: IntegrableKernel):
        """
        :param gpy_model: GPy.GPRegression model
        :param kernel: a kernel of type IntegrableKernel
        """

        self.gpy_model = gpy_model
        self.kern = kernel

    @property
    def X(self):
        return self.gpy_model.X

    @property
    def Y(self):
        return self.gpy_model.Y

    @property
    def noise_variance(self):
        """
        Gaussian observation noise variance
        :return: The noise variance from some external GP model
        """
        return self.gpy_model.Gaussian_noise[0]

    def update_data(self, X, Y):
        """
        Updates model with new training data
        :param X: New training features
        :param Y: New training outputs
        """
        self.gpy_model.set_XY(X, Y)

    def predict(self, X_pred, full_cov=False):
        """
        Predictive mean and (co)variance at the locations X_pred

        :param X_pred: points at which to predict, with shape (number of points, dimension)
        :param full_cov: if True, return the full covariance matrix instead of just the variance
        :return: Predictive mean, predictive (co)variance
        """
        return self.gpy_model.predict(X_pred, full_cov)

    def gram_chol(self):
        """
        The lower triangular cholesky decomposition of the kernel Gram matrix

        :return: a lower triangular matrix being the cholesky matrix of the kernel Gram matrix
        """
        return self.gpy_model.posterior.woodbury_chol

    def graminv_residual(self):
        """
        The inverse Gram matrix multiplied with the mean-corrected data

        ..math::

            (K_{XX} + \sigma^2 I)^{-1} (Y - m(X))

        where the data is given by {X, Y} and m is the prior mean

        :return: the inverse Gram matrix multiplied with the mean-corrected data with shape: (number of datapoints, 1)
        """
        return self.gpy_model.posterior.woodbury_vector

    def optimize(self):
        """ Optimize the hyperparameters of the model """
        self.gpy_model.optimize()

    def optimize_restarts(self, n_restarts):
        """
        Optimize the hyperparameters of the model with restarts

        :param n_restarts: Number of restarts
        """
        self.gpy_model.optimize_restarts(n_restarts)


class RBFGPy(IRBF):
    """
    Wrapper around the GPy RBF kernel
    """

    def __init__(self, gpy_rbf):
        """
        :param gpy_rbf: An RBF kernel from GPy
        """
        self.gpy_rbf = gpy_rbf

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


def gpy_wrapper_for_quadrature(gpy_model, integral_bounds):
    """
    Wrap a GPy model and return an emukit model

    :param gpy_model: A GPy Gaussian process regression model
    :param integral_bounds: Integration bounds (MultiDimensionalContinuousParameter)

    :return: emukit model for quadrature (IBaseGaussianProcess)
    """

    if gpy_model.kern.name is not 'rbf':
        """ currently only the RBF kernel is implemented """
        raise NotImplementedError

    # wrap the kernel
    rbf_kernel = RBFGPy(gpy_model.kern)
    integrable_rbf = IntegrableRBF(rbf_kernel=rbf_kernel,
                                   input_dim=gpy_model.X.shape[-1], integral_bounds=integral_bounds)

    # wrap the model
    return GPRegressionGPy(gpy_model, integrable_rbf)