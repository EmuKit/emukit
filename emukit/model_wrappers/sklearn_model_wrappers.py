import numpy
from typing import Tuple

from emukit.core.interfaces.models import IModel
from sklearn.gaussian_process import GaussianProcessRegressor

class sklearnGPRWrapper(GaussianProcessRegressor, IModel):

    def predict(self, X: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Predict mean and variance values for given points
        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        
        """
        return super(sklearnGPRWrapper,self).predict(X, return_cov=True)

    
    def set_data(self, X: numpy.ndarray, Y: numpy.ndarray) -> None:
        """
        Sets training data in model
        :param X: new points
        :param Y: function values at new points X
        
        """
        self.X_train_, self.y_train_ = X, Y
        
        
    def optimize(self) -> None:
        """
        Optimize hyper-parameters of model
        
        sklearn does the optimization inside the fit() method:
        https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/gaussian_process/_gpr.py#L222
        
        """
        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta,
                                                         clone_kernel=False)

            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel_.theta,
                                                      self.kernel_.bounds))]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not numpy.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[numpy.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -numpy.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta,
                                             clone_kernel=False)

        
        
        
    @property
    def X(self):
        return self.X_train_

    
    @property
    def Y(self):
        return self.y_train_
