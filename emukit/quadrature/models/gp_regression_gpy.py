import GPy
from ..kernels import IntegrableKernel
from ..methods import IBaseGaussianProcess


class GPRegressionGPy(IBaseGaussianProcess):
    """
    Wrapper for GPy GPRegression

    An instance of this can be passed as 'base_gp' to an ApproximateWarpedGPSurrogate object
    """
    def __init__(self, X, Y, kernel: IntegrableKernel, noise_variance=1.):
        """
        :param X: data points
        :param Y: function evaluations at X
        :param kernel: a kernel of type IntegrableKernel
        :param noise_variance: the Gaussian observation noise on Y
        """

        super(GPRegressionGPy, self).__init__(X, Y, kernel)
        self.gpy_model = GPy.models.GPRegression(X, Y, kernel.rbf.gpy_rbf, noise_var=noise_variance) #TODO: make it work with prior mean

    @property
    def noise_variance(self):
        """
        Gaussian observation noise variance
        :return: The noise variance from some external GP model
        """
        return self.gpy_model.Gaussian_noise[0]

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

    # non-required but useful wrappers
    def optimize(self):
        self.gpy_model.optimize()

    def optimize_restarts(self, n_restarts):
        self.gpy_model.optimize_restarts(n_restarts)