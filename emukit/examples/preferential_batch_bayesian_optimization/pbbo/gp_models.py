import numpy as np

import GPy
import itertools

from GPy.likelihoods import Gaussian
from GPy.core.parameterization.param import Param
from GPy.util import choleskies

from GPy.util.linalg import dtrtrs, dpotrs, pdinv, tdot, jitchol

import paramz
import os
import emukit

import traceback

from copy import deepcopy

from typing import Tuple, List, Callable, Dict

from .inferences import vi_batch_comparison as vi
from .inferences import ep_batch_comparison as ep
import stan_utility
from .inferences import StanPosterior
from . import util
from emukit.core.interfaces import IModel
import os


        
class ComparisonGP(GPy.core.Model):
    """
    A class for all common methods needed for the different ComparisonGP wrappers
    """
    def get_current_best(self) -> float:
        """
        :return: minimum of means of predictions at all input locations (needed by q-EI) 
        """
        return min(self.Y)

    def get_y_pred(self) -> np.ndarray:
        """
        :return: GP mean at inputs used to compute the posterior approximation (needed by q-EI)
        """
        y_pred, _ = self.predict(self.X, include_likelihood=False)
        return y_pred

    def log_likelihood(self) -> float:
        """
        :return: log marginal likelihood needed for optimizing hyper parameters and performing model comparison
        """
        return self._log_marginal_likelihood

    def predict(self, Xnew: np.ndarray, full_cov: bool=False, include_likelihood=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predictive mean and covariance of the GP at the input location
        :param Xnew: Locations the prections are wanted at
        :param full_cov: If the user wants the function to return the full covariance or only the diagonal
        :param include_likelihood: If the user wants the function to add the noise of the observations to the prediction covariance
        :return: predictive mean and predictive covariance
        """
        pred_mean, pred_var = self.posterior._raw_predict(self.kern, Xnew, self.X, full_cov=full_cov) #self.posterior._raw_predict(self.kern, np.hstack([Xnew,ki]), np.hstack([self.X, self.ki]), full_cov=full_cov)
        if include_likelihood:
            pred_var = pred_var + self.likelihood.variance
        return pred_mean, pred_var

    def predict_noiseless(self, Xnew, full_cov=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predictive mean and covariance of the GP latent function at the input location
        :param Xnew: Locations the prections are wanted at
        :param full_cov: If the user wants the function to return the full covariance or only the diagonal
        :return: predictive latent mean and predictive latent covariance
        """
        return self.predict(Xnew, full_cov=full_cov, include_likelihood=False)

    def posterior_samples_f(self, X: np.ndarray, size: int=10, **predict_kwargs) -> np.ndarray:
        """
        Draw random samples from the posterior predictive distribution
        
        :param X: Locations where the posterior samples should be drawn at
        :param size: Number of posterior samples
        :return: Simulated posterior samples
        """
        predict_kwargs["full_cov"] = True  # Always use the full covariance for posterior samples.
        predict_kwargs["include_likelihood"] = False
        m, v = self.predict(X,  **predict_kwargs)

        def sim_one_dim(m, v):
            # Draw posterior sample in one dimension
            return np.random.multivariate_normal(m, v, size).T

        if self.output_dim == 1:
            return sim_one_dim(m.flatten(), v)[:, np.newaxis, :]
        else:
            fsim = np.empty((X.shape[0], self.output_dim, size))
            for d in range(self.output_dim):
                if v.ndim == 3:
                    fsim[:, d, :] = sim_one_dim(m[:, d], v[:, :, d])
                else:
                    fsim[:, d, :] = sim_one_dim(m[:, d], v)
        return fsim


class EPComparisonGP(ComparisonGP):
    """
    GPy wrapper for a GP model consisting of preferential batch observations when the posterior is approximated using Expectation Propagation

    :param X: All locations of both direct observations and batch comparisons
    :param y: Direct observations in as a list of tuples telling location index (row in X) and observation value.
    :param yc: Batch comparisons in a list of lists of tuples. Each batch is a list and tuples tell the comparisons (winner index, loser index)
    :param kernel: A GPy kernel used 
    :param likelihood: A GPy likelihood. Only Gaussian likelihoods are accepted
    :param name: Name of the model. Defaults to 'EpComparisonGP'
    :param ep_max_it: Maximum number of iterations used when approximating the posterior in EP.
    :param eta: parameter for fractional EP updates.
    :param delta: damping EP updates factor.
    :param get_logger: Function for receiving the legger where the prints are forwarded.
    """
    def __init__(self, X: np.ndarray, y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]],
                 kernel: GPy.kern.Kern, likelihood: GPy.likelihoods.Gaussian, name: str='EPComparisonGP',
                 ep_max_itt: int=100, delta: float=0.5, eta: float=0.5, get_logger: Callable=None):

        super(EPComparisonGP, self).__init__(name=name)
        
        self.N, self.D = X.shape[0], X.shape[1]

        self.output_dim = 1 # hard coded, the code doesn't support multi output case
        
        self.X = X
        self.y = y
        self.yc = yc

        self.kern = kernel
        self.likelihood = likelihood 

        # A helper parameter for EP. Each observation could possibly come from different kernels and likelihoods.
        # The inference supports this already, but this GPy wrapper doesn't
        self.sigma2s = self.likelihood.variance*np.ones((X.shape[0],1), dtype=int)
        
        self.ep_max_itt = ep_max_itt
        self.eta = eta
        self.delta = delta
        
        self.link_parameter(self.kern)
        self.link_parameter(self.likelihood)
        self.posterior, self.ga_approx, self.Y = None, None, None
        self.get_logger = get_logger

    def parameters_changed(self):
        """
        Update the posterior approximation after kernel or likelihood parameters have changed or there are new observations
        """
        # Recompute the posterior approximation
        self.posterior, self._log_marginal_likelihood, self.grad_dict, self.ga_approx = ep.ep_comparison(self.X, self.y, self.yc,  self.kern, self.sigma2s,\
            max_itt=self.ep_max_itt, tol=1e-6, delta=self.delta, eta=self.eta, ga_approx_old=self.ga_approx, get_logger=self.get_logger)
        
        #predict Y at inputs (needed by q-EI)
        self.Y = self.get_y_pred()


    def set_XY(self, X: np.ndarray, y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]]):
        """
        Set new observations and recompute the posterior

        :param X: All locations of both direct observations and batch comparisons
        :param y: Direct observations in as a list of tuples telling location index (row in X) and observation value.
        :param yc: Batch comparisons in a list of lists of tuples. Each batch is a list and tuples tell the comparisons (winner index, loser index)
        """
        self.X = X
        self.y = y
        self.yc = yc
        self.sigma2s = self.likelihood.variance*np.ones((X.shape[0],1), dtype=int)
        self.parameters_changed()

class VIComparisonGP(ComparisonGP):
    """
    GPy wrapper for a GP model consisting of preferential batch observations when the posterior is approximated using Variational Inference

    :param X: All locations of both direct observations and batch comparisons
    :param y: Direct observations in as a list of tuples telling location index (row in X) and observation value.
    :param yc: Batch comparisons in a list of lists of tuples. Each batch is a list and tuples tell the comparisons (winner index, loser index)
    :param kernel: A GPy kernel used 
    :param likelihood: A GPy likelihood. Only Gaussian likelihoods are accepted
    :param vi_mode: A string indicating if to use full rank or mean field VI
    :param name: Name of the model. Defaults to 'VIComparisonGP'
    :param max_iters: Maximum number of iterations used when approximating the posterior in VI.
    :param get_logger: Function for receiving the legger where the prints are forwarded.
    """
    def __init__(self, X: np.ndarray, y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]],
                 kernel: GPy.kern.Kern, likelihood: Gaussian, vi_mode: str='fr',
                 name: str='VIComparisonGP', max_iters: int=50, get_logger: Callable=None):
        super(VIComparisonGP, self).__init__(name=name)

        self.N, self.D = X.shape[0], X.shape[1]

        self.output_dim = 1 
        self.get_logger = get_logger
        self.X = X
        self.y = y
        self.yc = yc

        self.max_iters = max_iters
        self.vi_mode = vi_mode

        self.kern = kernel
        self.likelihood = likelihood


        self.sigma2s = self.likelihood.variance * np.ones((X.shape[0], 1),
                                                          dtype=int)
        jitter = 1e-6
        K = self.kern.K(X)
        L = np.linalg.cholesky(K + np.identity(K.shape[0])*jitter)

        self.alpha = np.zeros((self.N,1))
        self.beta = np.ones((self.N,1))
        
        self.posterior = None
        
        # If we are using full rank VI, we initialize it with mean field VI
        if self.vi_mode == 'FRVI':
            self.posterior, _, _, self.alpha, self.beta = vi.vi_comparison(self.X, self.y, self.yc,
                                                                  self.kern, self.sigma2s,
                                                                  self.alpha, self.beta,
                                                                  max_iters=50, method='mf')
            self.beta = choleskies._triang_to_flat_pure(jitchol(self.posterior.covariance)[None,:])

    def parameters_changed(self):
        """
        Update the posterior approximation after kernel or likelihood parameters have changed or there are new observations
        """
        if self.vi_mode == 'fr':
            method = 'fr'
        else:
            method = 'mf'

        self.posterior, self._log_marginal_likelihood, self.grad_dict, alpha, beta = \
            vi.vi_comparison(self.X, self.y, self.yc, self.kern, self.sigma2s, self.alpha, self.beta,
                             max_iters=self.max_iters, method=method, get_logger=self.get_logger)

        self.alpha = alpha
        self.beta = beta
        self.Y = self.get_y_pred()

    def set_XY(self, X: np.ndarray, y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]]):
        """
        Set new observations and recompute the posterior

        :param X: All locations of both direct observations and batch comparisons
        :param y: Direct observations in as a list of tuples telling location index (row in X) and observation value.
        :param yc: Batch comparisons in a list of lists of tuples. Each batch is a list and tuples tell the comparisons (winner index, loser index)
        """
        self.N, self.D = X.shape[0], X.shape[1]
        self.X = X
        self.y = y
        self.yc = yc
        self.sigma2s = self.likelihood.variance * np.ones((X.shape[0], 1), dtype=int)
        alpha = np.zeros((self.N,1))
        if self.vi_mode == 'fr':
            beta = np.ones((int((self.N**2+ self.N)/2), 1))
        else:
            beta = np.ones((self.N, 1))
        for i in range(len(self.alpha)):
            alpha[i] = self.alpha[i]
        for i in range(len(self.beta)):
            beta[i] = self.beta[i]
        self.alpha, self.beta =alpha, beta
        self.posterior = None
        self.parameters_changed()

class VIComparisonGPMF(VIComparisonGP):
    """
    A very thin wrapper for mean field VI
    """
    def __init__(self, X: np.ndarray, y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]],
                 kernel: GPy.kern.Kern, likelihood: Gaussian, name: str='VIComparisonGPMF',
                 max_iters: int=50, get_logger: Callable=None):
        super(VIComparisonGPMF, self).__init__(X, y, yc, kernel, likelihood, name=name,
                                               max_iters=max_iters, vi_mode='mf', get_logger=get_logger)

class VIComparisonGPFR(VIComparisonGP):
    """
    A very thin wrapper for full rank VI
    """
    def __init__(self, X: np.ndarray, y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]],
                 kernel: GPy.kern.Kern, likelihood: Gaussian, name: str='VIComparisonGPFR',
                 max_iters: int=50, get_logger: Callable=None):
        super(VIComparisonGPFR, self).__init__(X, y, yc, kernel, likelihood, name=name,
                                               max_iters=max_iters, vi_mode='fr', get_logger=get_logger)
        
        
class MCMCComparisonGP(ComparisonGP):
    """
    GPy wrapper for a GP model consisting of preferential batch observations when the posterior is approximated using MCMC smaples

    :param X: All locations of both direct observations and batch comparisons
    :param y: Direct observations in as a list of tuples telling location index (row in X) and observation value.
    :param yc: Batch comparisons in a list of lists of tuples. Each batch is a list and tuples tell the comparisons (winner index, loser index)
    :param kernel: A GPy kernel used 
    :param likelihood: A GPy likelihood. Only Gaussian likelihoods are accepted
    :param posterior_samples: Number of posterior samples used for approximation
    :param vi_mode: A string indicating if to use full rank or mean field VI
    :param get_logger: Function for receiving the legger where the prints are forwarded.
    """
    def __init__(self, X: np.ndarray, y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]],
                 kernel: GPy.kern.Kern, likelihood: Gaussian, posterior_samples: int=45,
                 get_logger: Callable=None):
        super(MCMCComparisonGP, self).__init__(name="MCMC")
        self.N, self.D = X.shape[0], X.shape[1]
        self.output_dim = 1 ## hard coded
        self.posterior_samples = posterior_samples
        self.X = X
        self.y = y
        self.yc = yc
        self.noise_std= np.sqrt(likelihood.variance[0])
        self.sigma2s = (likelihood.variance[0])*np.ones((X.shape[0],1), dtype=int)
        self.variance = kernel.variance[0]
        self.lengthscale = np.array([kernel.lengthscale[:]]).flatten()
        self.kern = kernel
        self.posterior = None
        if not os.path.exists(stan_utility.file_utils. get_path_of_cache()):
            os.makedirs(stan_utility.file_utils. get_path_of_cache())
        self.model = stan_utility.compile_model(os.path.join(os.path.dirname(__file__), "inferences/sexpgp_comparison.stan"), model_name='comparison_model')
        self.get_logger = get_logger
        self.parameters_changed()

    def _fit_stan(self) -> Dict:
        """
        Fit stan using the given observations and return a wrapper object that allows a simple framework for handling the posterior samples
        
        :return: A pystan fit object containing the posterior fit for the model
        """
        def inits():
            return {"eta": np.zeros((self.X.shape[0],))}
        dat = {'N': len(self.y), 'N_comp': len(list(itertools.chain(*self.yc))), 'Nf': self.X.shape[0], 'd': self.X.shape[1],
        'y': np.array([yi[1] for yi in self.y]).flatten(),
        'yi': np.array([yi[0]+1 for yi in self.y],dtype=int),
        'y_comp1': np.array([i[0]+1 for i in list(itertools.chain(*self.yc))],dtype=int),
        'y_comp2': np.array([i[1]+1 for i in list(itertools.chain(*self.yc))], dtype=int),
        'x':  self.X,
        'rho': self.lengthscale, 'alpha':np.sqrt(self.variance) , 'sigma': self.noise_std ,'delta': 1e-9}
        iter = 3000
        chains = 6
        warmup = iter - int(self.posterior_samples // chains)
        if self.get_logger is not None:
            self.get_logger().info("Starting to sample from the posterior")
        fit = self.model.sampling(data=dat, seed=194838, chains=6, n_jobs=1, iter=iter, warmup=warmup, refresh=-1, control={'adapt_delta': 0.9999, 'max_treedepth':12}, init=inits)
        if self.get_logger is not None:
            self.get_logger().info("Sampling finished")
        return fit

    def parameters_changed(self):
        """
        Update the posterior approximation after kernel or likelihood parameters have changed or there are new observations
        """
        # Run Stan:
        fit = self._fit_stan() 
        self.samples = fit.extract(permuted=True)
        remove_levels = True
        if len(self.y) > 0:
            remove_levels=False
        self.posterior = StanPosterior(self.samples, self.X, self.kern, self.noise_std, Y = np.array([yi[1] for yi in self.y]).flatten(), remove_levels=remove_levels, y=self.y, yc=self.yc, get_logger=self.get_logger)
        self._log_marginal_likelihood, self.grad_dict = None, {}
        self.y_pred = self.get_y_pred()
        self.Y = self.y_pred

    def set_XY(self, X: np.ndarray, y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]]):
        """
        Set new observations and recompute the posterior

        :param X: All locations of both direct observations and batch comparisons
        :param y: Direct observations in as a list of tuples telling location index (row in X) and observation value.
        :param yc: Batch comparisons in a list of lists of tuples. Each batch is a list and tuples tell the comparisons (winner index, loser index)
        """
        self.X = X
        self.y = y
        self.yc = yc
        self.parameters_changed()

class ComparisonGPEmukitWrapper(IModel):
    """
    This is a thin wrapper around ComparisonGP to allow users to plug GPy models into Emukit
    """
    def __init__(self, gpy_model: ComparisonGP, batch_size=1):
        """
        :param gpy_model: GPy model object to wrap
        :param batch_size: Number of samples in a batch
        """
        self.model = gpy_model
        self.batch_size = batch_size
        self.g = 0

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 of the predictive distribution at each input location
        """
        return self.model.predict(X)

    def predict_with_full_covariance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 and n_points x n_points of the predictive
                 mean and variance at each input location
        """
        return self.model.predict(X, full_cov=True)

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model

        :param X: New training features
        :param Y: New training outputs
        """
        if X.shape[0]>0:
            X = X[0,:].reshape((1,-1))
            Y = Y[0,:].reshape((-1))
            ind = self.model.X.shape[0]
            X_set = self.model.X
            y_set = self.model.y
            yc_set = self.model.yc
            if X.shape[1]==self.model.X.shape[1]: #Direct observation
                X_set = np.concatenate((self.model.X, X), axis=0)
                y_set += [(i+ind, Y[i]) for i in range(len(Y))]
            else:
                X_tmp = np.concatenate(tuple(X[:,i*self.batch_size:(i+1)*self.batch_size] for i in range(len(Y))), axis=0)
                X_set = np.concatenate((self.model.X, X_tmp), axis=0)
                yc_set += [ [ (yc[0]+ind, yc[1]+ind) for yc in util.comparison_form(Y)]]
        
            self.model.set_XY(X_set, y_set, yc_set)
        #
        #self.model.set_XY(X, Y)

    def predict_covariance(self, X: np.ndarray, with_noise: bool=True) -> np.ndarray:
        """
        Calculates posterior covariance between points in X
        :param X: Array of size n_points x n_dimensions containing input locations to compute posterior covariance at
        :param with_noise: Whether to include likelihood noise in the covariance matrix
        :return: Posterior covariance matrix of size n_points x n_points
        """
        _, v = self.model.predict(X, full_cov=True, include_likelihood=with_noise)
        v = np.clip(v, 1e-10, np.inf)

        return v

    @property
    def X(self) -> np.ndarray:
        """
        :return: An array of shape n_points x n_dimensions containing training inputs
        """
        return np.empty((0, self.model.X.shape[1]))

    @property
    def Y(self) -> np.ndarray:
        """
        :return: An array of shape n_points x 1 containing training outputs
        """
        return np.empty((0,1))
    
    def optimize(self):
        """
        Optimizes the model. In this case does nothing since the parameters are fixed
        """
        pass