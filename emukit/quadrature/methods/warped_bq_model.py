import numpy as np
from typing import Tuple

from emukit.core.interfaces.models import IModel
from ..interfaces.base_gp import IBaseGaussianProcess


class WarpedBayesianQuadratureModel(IModel):
    """
    The general class for Bayesian quadrature (BQ) with a warped Gaussian process.
    
    Inference is performed with the warped GP, but the integral is computed on a Gaussian approximation.
    The warping of the base GP is encoded in the methods 'transform' and 'inverse_transform'

    Examples of warping-approximation pairs:
    - a moment matched squared GP (wsabi-m)
    - a linear approximation to a squared GP (wsabi-l)
    - no approximation if there is no warping (Vanilla BQ)
    - ...
    """
    def __init__(self, base_gp: IBaseGaussianProcess) -> None:
        """    
        :param base_gp: the underlying GP model
        """
        self.base_gp = base_gp

    @property
    def X(self):
        return self.base_gp.X

    @property
    def Y(self):
        return self.base_gp.Y

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Transform from base-GP to integrand.
        """
        raise NotImplemented

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Transform from integrand to base-GP.
        """
        raise NotImplemented

    def predict_base(self, X_pred: np.ndarray, return_full_cov: bool) -> Tuple:
        """
        Computes predictive means and (co-)variances of the warped GP as well as the base GP

        :param X_pred: Locations at which to predict
        :param return_full_cov: If True, full covariance matrices will be returned. Otherwise variances only.

        :returns: predictive mean and (co-)variance of warped GP, predictive mean and (co-)variance of base-GP.
        """
        raise NotImplemented

    def predict_with_full_covariance(self, X_pred: np.ndarray) -> Tuple:
        """
        Computes approximate predictive means and full co-variances of warped GP.

        :param X_pred: Locations at which to predict
        :return: predictive mean, predictive full co-variance of warped-GP
        """
        return self.predict_base(X_pred, return_full_cov=True)[:2]

    def predict(self, X_pred: np.ndarray) -> Tuple:
        """
        Computes predictive means and variances of warped-GP.

        :param X_pred: Locations at which to predict
        :return: predictive mean, predictive variances of warped-GP
        """
        return self.predict_base(X_pred, return_full_cov=False)[:2]

    def set_data(self, X: np.ndarray, Y: np.ndarray):
        self.base_gp.set_data(X, Y)

    def optimize(self):
        self.base_gp.optimize()

    def integrate(self) -> Tuple[float, float]:
        """
        Computes an estimator of the integral as well as its variance. 

        :returns: estimator of integral and its variance
        """
        raise NotImplemented

