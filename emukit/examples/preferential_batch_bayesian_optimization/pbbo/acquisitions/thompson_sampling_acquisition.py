from .. import util
from .. import ComparisonGP, ComparisonGPEmukitWrapper
from .acquisition_function import AcquisitionFunction
import GPy
import numpy as np
import numpy.linalg as la
import scipy as sp
import collections
import time
from scipy.stats import norm, mvn, multivariate_normal
from GPy.util.univariate_Gaussian import std_norm_pdf, std_norm_cdf, derivLogCdfNormal, logCdfNormal, cdfNormal

from typing import Callable, List, Tuple, Dict

class SequentialThompsonSampler():
    """
    A class for performing Thompson sampling from a GP. The sampler works so that
    it stores the simulated draws and conditions the new draws on (part of) them.
    
    :param model: A model from which posterior the samples are drawn from
    :param seed: Random seed that specifies the random sample
    :param delta: Defines how close the samples for numerical derivatives are taken from
    :param num_points: Number of points the samples are conditioned on
    """
    def __init__(self, model: ComparisonGP, seed: float=None, delta: float=1e-5, num_points: int=None):
        self.model = model #model
        self.posterior = model.posterior

        self.d = self.model.X.shape[1] #dimensionality of the input points

        self.seeds = []
        self.reset(seed=seed)

        self.delta = delta
        if num_points is None:
            num_points = int(100 * round(np.sqrt(self.d)))
        self.num_points = num_points
        self.scaling = np.array([model.kern.lengthscale[:]]).flatten()

    def reset(self, seed: float=None) -> None:
        """
        Reset the sampler by forgetting the already drawn samples and resetting the seed.
        
        :param seed: new seed after reset
        """
        self.x_evaluations = np.empty((0,self.model.X.shape[1]))
        self.f_evaluations = np.empty((0,1))
        if seed is None:
            max = 1000000
            if (self.model.name == "MCMC"):
                max = self.model.posterior.samples['f'].shape[0]
            seed = np.random.randint(max)
            #Check if we already have used this seed
            while (seed in self.seeds):
                seed = np.random.randint(max)
        self.seeds = self.seeds + [seed]
        np.random.seed(seed)

    def add_data(self, x: np.ndarray, f: np.ndarray) -> None:
        """
        Add new points and evaluations to the stack of already evaluated points
        
        :param x: new locations
        :param f: new evaluations
        """
        self.x_evaluations = np.concatenate((self.x_evaluations, x), axis=0)
        self.f_evaluations = np.concatenate((self.f_evaluations, f), axis=0)

    def get_mu_sigma(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the posterior predictive distribution at the given locations without conditionin on the already evaluated locatons
        
        :param x: New locations where the distribution is wanted at
        :return: Tuple containing the predictive mean and covariance
        """
        if (self.model.name == "MCMC"):
            mu, L = self.posterior._get_mu_L(x, with_index=self.seeds[-1])
            Sigma = L[0,:,:] @ L[0,:,:].T
        else:
            mu, Sigma = self.model.predict_noiseless(x, full_cov=True)
            if (len(Sigma.shape) > 2):
                Sigma = Sigma[:,:,0]
        mu = mu.reshape(-1, 1)
        Sigma = Sigma + 1e-7*np.eye(Sigma.shape[0]) #noise for robustness
        return mu, Sigma

    def get_posterior_predictive_distribution(self, x: np.ndarray, inds: int=np.arange(0)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the predictive distribution at the given locations by conditionin on old points and the posterior
        
        :param x: New location where the distribution is wanted at
        :param inds: indices of already evaluated points the distribution is conditioned with
        :return: Tuple containing the predictive mean and covariance
        """
        n1, n2 = inds.shape[0], x.shape[0]
        mu, Sigma = self.get_mu_sigma(np.concatenate((self.x_evaluations[inds,:].reshape(-1,self.d), x), axis=0))
        if(n1>0):
            kXX = Sigma[np.ix_(range(n1),range(n1))]
            kxX = Sigma[np.ix_(range(n1,n1+n2), range(n1))]
            kxx = Sigma[np.ix_(range(n1,n1+n2),range(n1,n1+n2))]
            kXX_inv = la.inv(kXX)
            mu_pred = mu[n1:] + kxX @ kXX_inv @ (self.f_evaluations[inds] - mu[:n1])
            Sigma_pred = kxx - kxX @ kXX_inv @ kxX.T
            return mu_pred, Sigma_pred
        else:
            return mu, Sigma

    def get_posterior_predictive_distribution_robust(self, x, num_points=100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the predictive distribution at the given locations by conditionin on old points and the posterior
        
        :param x: new point the sample is wanted to evaluate at
        :param num_points: number of points taken into account when computing the predictive distribution
        :return: Tuple containing the predictive mean and covariance
        """
        N,n = self.x_evaluations.shape[0], x.shape[0]
        inds = np.arange(N)
        if N >= num_points:
            inds = np.empty((num_points, n), dtype=int)
            for i in range(n):
                dists = np.linalg.norm((self.x_evaluations-x[None,i,:,])/self.scaling[None,:], axis=1)
                inds[:, i] = np.argsort(dists)[:num_points]
            inds = np.unique(inds.flatten())
        return self.get_posterior_predictive_distribution(x, inds=inds)

    def evaluate_and_add(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the value and gradient of the sample at given point
        
        :param x: new point the sample is wanted to evaluate at
        :return: Tuple containing the value and gradient
        """
        X = np.atleast_2d(X).reshape((-1,self.d))
        ff, fgrad = np.empty((0)), np.empty((0, self.d))
        for i in range(X.shape[0]):
            #deviate to compute the numerical gradients
            x = np.tile(X[i,:].reshape((1,self.d)), (self.d+1, 1))
            for i in range(self.d):
                x[i+1,i] = x[i+1,i] + self.delta
            # Get predictive distribution
            mu, k = self.get_posterior_predictive_distribution_robust(x, num_points = self.num_points)

            #evaluate and add
            f = np.random.multivariate_normal(mu.flatten(),k).reshape((-1, 1))
            self.add_data(x, f)

            #Evaluate numerical gradients
            grad = np.empty((1,self.d))
            for i in range(self.d):
                grad[0,i] = (f[i+1]-f[0])/self.delta
            ff = np.concatenate((ff, f[0]), axis=0)
            fgrad = np.concatenate((fgrad, grad), axis=0)
        return ff.reshape((-1,1)), fgrad



class ThompsonSampling(AcquisitionFunction):

    def acq_fun_optimizer(self, m: ComparisonGP, bounds: np.ndarray, batch_size: int, get_logger: Callable) -> np.ndarray:
        """
        Implements the optimization scheme for the Thompson sampling acquisition function
        
        :param m: The model which posterior is used by the acquisition function (from which the samples are drawn from)
        :param bounds: the optimization bounds of the new sample
        :param batch_size: How many points are there in the batch
        :param get_logger: Function for receiving the legger where the prints are forwarded.
        :return: optimized locations
        """
        self.sampler = SequentialThompsonSampler(m)
        X_all = np.empty((batch_size, m.X.shape[1]))
        for i in range(batch_size):
            self.sampler.reset() # reset the sampling seed
            yb = np.inf
            Xb = None
            for j in range(self.acq_opt_restarts):
                try:
                    # Initial point of the optimization
                    X0 = util.random_sample(bounds, 1)
                    res = sp.optimize.minimize(fun=self.sampler.evaluate_and_add,
                                                x0=X0.flatten(),
                                                method='L-BFGS-B',
                                                jac=True,
                                                bounds=bounds,
                                                options=self.optimizer_options
                                                )
                    X = res.x.reshape(1, len(bounds))
                    y = res.fun
                    if y < yb:
                        yb=y
                        Xb=X
                except Exception as e:
                    get_logger().error('Solver failed. Below the exception\n{}'.format(e))
            assert Xb is not None
            X_all[i,:] = Xb
        return X_all
    
    def reset(self, model: ComparisonGPEmukitWrapper) -> None:
        """
        Some acquisition functions need to be reseted, this method is for that.
        :param m: the GP model which posterior is used
        """
        try:
            self.sampler.reset()
        except:
            self.sampler = SequentialThompsonSampler(model)
            
    def evaluate(self, x: np.ndarray, model: ComparisonGPEmukitWrapper) -> np.ndarray:
        """
        Computes the Expected Improvement.
        :param x: points where the acquisition is evaluated.
        :param m: the GP model which posterior is used
        :return: acquisition function value
        """
        return self.sampler.evaluate_and_add(x)[0]

    def evaluate_with_gradients(self, x: np.ndarray, model: ComparisonGPEmukitWrapper) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Expected Improvement.
        :param x: points where the acquisition is evaluated.
        :param model: the GP model which posterior is used
        :return: A tuple containing the acquisition function values and their gradients
        """
        return self.sampler.evaluate_and_add(x)