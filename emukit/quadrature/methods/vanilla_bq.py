import numpy as np
from scipy.linalg import lapack

from emulab.quadrature.methods.warped_bq_model import WarpedBayesianQuadratureModel

class VanillaBayesianQuadrature(WarpedBayesianQuadratureModel):
    """
    class for vanilla Bayesian quadrature
    """

    def __init__(self, base_gp):
        """
        :param base_gp: a model derived from BaseGaussianProcess
        """
        super(VanillaBayesianQuadrature, self).__init__(base_gp)

    def transform(self, Y):
        """ Transform from base-GP to integrand """
        return Y

    def inverse_transform(self, Y):
        """ Transform from integrand to base-GP """
        return Y

    def predict(self, X_pred, return_full_cov=False):
        """ Computes the predictive mean and covariance """
        m, cov = self.base_gp.predict(X_pred, return_full_cov)
        return m, cov, m, cov

    def integrate(self):
        """
        Computes the estimator for the integral and a variance
        """
        return self._compute_integral_mean_and_variance()

    # helpers
    def _compute_integral_mean_and_kernel_mean(self):
        kernel_mean_X = self.base_gp.kern.qK(self.X)
        integral_mean = np.dot(kernel_mean_X, self.base_gp.graminv_residual())[0, 0]
        return integral_mean, kernel_mean_X

    def _compute_integral_mean_and_variance(self):
        integral_mean, kernel_mean_X = self._compute_integral_mean_and_kernel_mean()
        integral_var = self.base_gp.kern.qKq() - np.square(lapack.dtrtrs(self.base_gp.gram_chol(), kernel_mean_X.T,
                                                                       lower=1)[0]).sum(axis=0, keepdims=True).T
        return integral_mean, integral_var
