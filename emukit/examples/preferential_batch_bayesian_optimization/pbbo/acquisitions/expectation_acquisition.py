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
import matplotlib.pyplot as plt
import math
import time

from typing import Tuple, List, Callable, Dict


def dK_dX(self, X: np.ndarray, X2: np.ndarray, dimX: int):
    """
    Derivative of RBF kernel with respect to input X
    
    :param X: First input
    :param X2: Second input
    :param dimX: dimension of X the derivative is wanted at
    :return: Derivative of the kernel with respect to the wanted dimension of X
    """
    r = self._scaled_dist(X, X2)
    K = self.K_of_r(r)
    dist = X[:,None,dimX]-X2[None,:,dimX]
    lengthscale2inv = (np.ones((X.shape[1]))/(self.lengthscale**2))[dimX]
    return -1.*K*dist*lengthscale2inv
GPy.kern.RBF.dK_dX = dK_dX

#Some helper functions
phi = lambda x: norm.cdf(x) # cdfNormal(x)
npdf = lambda x, m, v: 1./np.sqrt(2*np.pi*v)*np.exp(-(x-m)**2/(2*v))

class ExpectationAcquisition(AcquisitionFunction):
    """
    Implements the general acquisition function that requires sampling from the posterior and uses reparametrization trick to compute the expectation required by the acquisition function
    """

    def dgp_ds_via_L(self, s: np.ndarray, dl: np.ndarray) -> np.ndarray:
        """
        Partial derivatives of the gp posterior samples with respect to the input locations given gradient with respect to cholesky the posterior covariance matrix.
        The derivative is computed using the generalised chain rule, meaning that derivative with respect to the posterior covariance matrix already exists  
        
        :param s: Samples of observations from the posterior distribution of the model
        :param dl: Derivatives of the acquisition with respect to the covariance matrix
        :return: the derivatives of the gp samples with respect to the inputs
        """
        dls = np.zeros((s.shape[0], dl.shape[1],dl.shape[3], dl.shape[4]))
        for i in range(dl.shape[3]):
            for j in range(dl.shape[4]):
                dls[:,:,i,j] = np.matmul(dl[:,:,:,i,j], s[:,:,None])[:,:,0]
        return dls

    def dgp_dL_via_Sigma(self, L: np.ndarray, L_inv: np.ndarray, dsigma: np.ndarray) -> np.ndarray:
        """
        Partial derivatives of the gp posterior samples with respect to the cholesky of the posterior covariance matrix given the partial derivative values with respect to the posterior covariance matrix.
        
        :param s: Samples of observations from the posterior distribution of the model
        :param L: Cholesky decomposition(s) of the posterior covariance matrix (samples)
        :param L_inv: Inverse(s) of Cholesky decomposition(s) of the posterior covariance matrix (samples)
        :param dsigma: Partial derivatives with respect to the posterior covariance matrix
        :return: the derivative of the gp samples with respect to the choleskies
        """
        E = np.tril(2*np.ones((L.shape[1],L.shape[2])),-1 ) +np.eye(L.shape[2])
        dl = np.empty(dsigma.shape) #np.empty((N,b,b,b,d))
        for i in range(dsigma.shape[3]):
            for j in range(dsigma.shape[4]):
                tmp1 = np.matmul(L_inv, dsigma[:,:,:,i,j]) # N x b x b
                tmp2 = np.matmul(tmp1, np.swapaxes(L_inv,1,2)) # N x b x b
                tmp3 = tmp2 * E[None,:,:] # N x b x b
                dl[:,:,:,i,j] = 0.5 * np.matmul(L, tmp3) # N x b x b
        return dl # N x b x b

    def dgp_dSigma(self, x: np.ndarray, X: np.ndarray, kern: GPy.kern.Kern, w_inv: np.ndarray) -> np.ndarray:
        """
        Partial derivatives of the gp posterior samples with respect to the posterior covariance matrix
        
        :param x: The locations the samples are taken at
        :param X: The locations used to train the GP model
        :param kern: Prior covariance matrix
        :param w_inv: inverses of the woodbury matrix of the model
        :return: the derivative of the gp samples with respect to the matrix
        """
        N, b, d, n = w_inv.shape[0], x.shape[0], x.shape[1], X.shape[0]
        dkxX_dx = np.empty((b,n,d))
        dkxx_dx = np.empty((b,b,d))
        for i in range(d):
            dkxX_dx[:,:,i] = kern.dK_dX(x, X, i)
            dkxx_dx[:,:,i] = kern.dK_dX(x, x, i)
        K = kern.K(x,X)

        grad = np.empty((N, b, d))
        dsigma = np.zeros((N,b,b,b,d))
        for i in range(b):
            for j in range(d):
                Ks = np.zeros((b, n))
                Ks[i,:] = dkxX_dx[i,:,j]
                dKss_dxi = np.zeros((b, b))
                dKss_dxi[i,:] = dkxx_dx[i,:,j]
                dKss_dxi[:,i] = dkxx_dx[i,:,j].T
                dKss_dxi[i,i] = 0
                dsigma[:,:,:,i,j] = dKss_dxi[None,:,:] - np.matmul(np.matmul(Ks[None,:,:], w_inv), (K.T)[None,:,:]) - np.matmul(np.matmul(K[None,:,:], w_inv), (Ks.T)[None,:,:])
        return dsigma

    def dgp_dmean(self, kern: GPy.kern.Kern, w_vec: np.ndarray, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Partial derivatives of the gp posterior samples with respect to the posterior mean
        
        :param kern: Prior covariance matrix
        :param w_vec: woodbury vectors of the posterior of the model
        :param x: The locations the samples are taken at
        :param X: The locations used to train the GP model
        :return: the derivative of the gp samples with respect to the mean
        """
        N, b, d, n = w_vec.shape[0], x.shape[0], x.shape[1], X.shape[0]
        dkxX_dx = np.empty((b,n,d))
        dmu = np.zeros((N,b,b,d))
        for i in range(d):
            dkxX_dx[:,:,i] = kern.dK_dX(x, X, i)
            for j in range(b):
                dmu[:,j,j,i] = np.matmul(dkxX_dx[j,:,i][None,:], w_vec[:,:,None]).flatten() # d
        return dmu #N x b x b x d

    def dgp_dx(self, s: np.ndarray, L: np.ndarray, L_inv: np.ndarray, w_vec: np.ndarray, w_inv: np.ndarray,
           kern: GPy.kern.Kern, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Partial derivatives of the acquisition function values with respect to the sampled observations from the GP using generalised chain rule.
        
        :param s: Samples of observation from the posterior distribution of the model
        :param L: Cholesky decomposition(s) of the posterior covariance matrix (samples)
        :param L_inv: Inverse(s) of Cholesky decomposition(s) of the posterior covariance matrix (samples)
        :param w_vec: woodbury vectors of the posterior of the model
        :param w_inv: inverses of the woodbury matrix of the model
        :param kern: Prior covariance matrix
        :param x: The locations the samples are taken at
        :param X: The locations used to train the GP model
        :return: the derivative of the acquisition function values with respect to the samples
        """
        dmu_dx = self.dgp_dmean(kern, w_vec, x, X)
        dsigma_dx = self.dgp_dSigma(x,X,kern,w_inv) # N x b x b x b x d
        dl_dx = self.dgp_dL_via_Sigma(L, L_inv, dsigma_dx) # N x b x b x b x d
        dls_dx = self.dgp_ds_via_L(s, dl_dx) # N x b x b x d
        return dmu_dx + dls_dx #N x b x b x d

    def acquisition_fun(self, x: np.ndarray, m: ComparisonGP) -> Tuple[float, np.ndarray]:
        """
        Computes the acquisition function value and acquisition function derivatives at the input locations

        :param x: input location the acquisition and its derivative are evaluated at
        :param m: the GP model which posterior is used
        :return: Tuple containing the acquisition function value and its gradients
        """
        pools = self.pool_size
        N= self.acq_samples
        if  m.name != "MCMC":
            pools = 1
        else:
            if pools != -1:
                N = N // pools
                Np = 1
            else:
                pools = 1
                Np = N


        x = np.array(x)
        #Assign values as class variables so there is no need to pass them
        b, d, n = x.shape[0], x.shape[1], m.X.shape[0] #size of the batch, dimensions of x, number of samples in the training data

        opt_val = 0
        grad = np.zeros((b,d))
        for i in range(pools):
            #Take model specific variables
            if(m.name == "MCMC"):
                #We take predictive mu and L (Sigma = L L^T) of the posterior
                mu, L, w_inv, w_vec = m.posterior._get_mu_L(x, N=Np, woodbury_inv=True)
                L_inv = np.empty(L.shape)
                for i in range(L.shape[0]):
                    L_inv[i,:,:] = la.inv(L[i,:,:])
            else:
                mu, Sigma = m.predict_noiseless(x,full_cov=True)
                if len(Sigma.shape)==3:
                    Sigma=Sigma[:,:,0]
                L = la.cholesky(Sigma)
                L_inv = la.inv(L)
                mu, L, L_inv = mu[None,:,0], L[None,:,:], L_inv[None,:,:]
                w_inv = m.posterior.woodbury_inv[None,:,:]
                w_vec = m.posterior.woodbury_vector[None,:,0]
                if(len(w_inv.shape)>3):
                    w_inv = w_inv[:,:,:,0]

            #Sample from N(0,I)
            s = np.random.normal(size=(N, b))
            s2 = np.random.normal(size=(N, b))
            #Generate samples for stochastic estimation of the expectation from the
            f = mu + np.matmul(L, s[:,:,None])[:,:,0]
            y = f + np.sqrt(m.sigma2s[0])*s2[:,:]

            sqrt_nu = np.sqrt(m.sigma2s[0]*np.ones(b))

            #Compute acquisition function value
            opt_val += self.opt_val(f,y,m,sqrt_nu)

            grad += self.opt_val_grad(s, f, y, m, sqrt_nu, L, L_inv, w_vec, w_inv, x)

        return -opt_val/pools, -grad/pools

    def acquisition_fun_flat(self, X: np.ndarray, m: ComparisonGP) -> Tuple[float, np.ndarray]:
        """
        Wrapper for acquisition_fun, where X is considered as a vector
        
        :param X: input location the acquisition and its derivative are evaluated at
        :param m: the GP model which posterior is used
        :return: Tuple containing the acquisition function value and its gradients
        """
        X = np.atleast_2d(X) 
        n = m.X.shape[1]
        k = X.shape[1] // n
        opt_val = []
        dpdX = np.empty((0, n*k))
        for i in range(X.shape[0]):
            X_i = X[i,:]
            start = time.time()
            (opt_val_i, dpdX_i) = self.acquisition_fun(X_i.reshape(k, n), m)
            opt_val += [opt_val_i]
            dpdX = np.concatenate((dpdX, dpdX_i.flatten().reshape((1,-1))), axis=0)
        return np.array(opt_val).reshape((-1,1)), dpdX

    def acq_fun_optimizer(self, m: ComparisonGP, bounds: np.ndarray, batch_size: int, get_logger: Callable) -> np.ndarray:
        """
        Implements the optimization scheme for the sampling based acquisition function
        
        :param m: The model which posterior is used by the acquisition function (from which the samples are drawn from)
        :param bounds: the optimization bounds of the new sample
        :param batch_size: How many points are there in the batch
        :param get_logger: Function for receiving the legger where the prints are forwarded.
        :return: optimized locations
        """
        
        X = None    # Will hold the final choice
        y = None    # Will hold the expected improvement of the final choice

        # Run local gradient-descent optimizer multiple times
        # to avoid getting stuck in a poor local optimum
        # Tile bounds to match batch size
        # print(self.batch_size)
        
        bounds_tiled = np.tile(bounds, (batch_size, 1))

        xss = []
        for j in range(self.acq_opt_restarts):
            # Initial point of the optimization
            X0 = util.random_sample(bounds, batch_size)
            get_logger().info("acq opt restart {}".format(j))
            try:
                f_, _ = self.acquisition_fun_flat(X0.flatten(), m)
                get_logger().debug("Starting from: {} with acq {}".format(X0.flatten(), f_))
                res = sp.optimize.minimize(fun=self.acquisition_fun_flat,
                                            x0=X0.flatten(),
                                            args=(m),
                                            method='L-BFGS-B',
                                            jac=True,
                                            bounds=bounds_tiled,
                                            options=self.optimizer_options
                                            )
                X0 = res.x.reshape(batch_size, len(bounds))
                y0 = res.fun
                get_logger().debug("Ended to: {} with acq {}".format(X0, y0))
                # Update X if the current local minimum is
                # the best one found so far
                X0 = X0.reshape(batch_size, len(bounds))
                if X is None or y0 < y:
                    X = X0
                    y = y0
            except Exception as e:
                get_logger().error('Solver failed. Below the exception\n{}'.format(e))
        # Assert that at least one optimization run succesfully
        assert X is not None
        return X
    
    def evaluate(self, x: np.ndarray, model: ComparisonGPEmukitWrapper) -> np.ndarray:
        """
        Computes the acquisition function at a batch.
        :param x: points where the acquisition is evaluated.
        :param batch_size: How many points are there in the batch.
        :return: acquisition function value evaluated at the batch
        """
        return self.acquisition_fun_flat(x, model.model)[0]
    
    def evaluate_with_gradients(self, x: np.ndarray, model: ComparisonGPEmukitWrapper) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the acquisition function at a batch.
        :param x: points where the acquisition is evaluated.
        :param batch_size: How many points are there in the batch.
        :return: A tuple containing acquisition function value evaluated at the batch and its gradient
        """
        return self.acquisition_fun_flat(x, model.model)

class SumOfVariances(ExpectationAcquisition):
    """
    This class implements the Sum of Variances acquisition function
    """
    
    def opt_val(self, f: np.ndarray, y: np.ndarray, m: ComparisonGP, sqrt_nu: np.ndarray) -> float:
        """
        computes the acquisition function value by sampling
        
        :param f: latent function samples of the posterior
        :param y: observations from the latent functio (latent function values corrupted with noise)
        :param m: the model which posterior is used
        :param sqrt_nu: noise std for the observations
        :return: Sum of variances at the input
        """
        liks = np.zeros(f.shape[1])
        for i in range(f.shape[1]): #Loop through each member being the batch winner (lower value better)
            mi = [k for k in range(f.shape[1]) if k is not i] # select all indices but the batch winner
            prs = np.prod(phi((f[:, mi] - y[:,i, None])/sqrt_nu[None,mi]), axis=1) #Equation 3 in the draft (or the simplified version of it at the end of Section 2.2)
            prs2 = prs**2
            liks[i] = np.mean(prs2,axis=0) - np.mean(prs, axis=0)**2 #Variance of p(y_i \leq y_j \forall i \neq j) weighted by p(y_i \leq y_j \forall i \neq j)
        return np.sum(liks)

    def opt_val_grad(self, s: np.ndarray, f: np.ndarray, y: np.ndarray, m: ComparisonGP, sqrt_nu: np.ndarray, L: np.ndarray, L_inv: np.ndarray, w_vec: np.ndarray, w_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        computes the acquisition function value by sampling
        
        :param f: latent function samples of the posterior
        :param y: observations from the latent functio (latent function values corrupted with noise)
        :param m: the model which posterior is used
        :param sqrt_nu: noise std for the observations
        :param L: Cholesky decomposition(s) of the posterior covariance matrix (samples)
        :param L_inv: Inverse(s) of Cholesky decomposition(s) of the posterior covariance matrix (samples)
        :param w_vec: woodbury vectors of the posterior of the model
        :param w_inv: inverses of the woodbury matrix of the model
        :param x: The locations the samples are taken at      
        :return: Gradient of Sum of variances at the input
        """
        
        N, b, d, n = f.shape[0], x.shape[0], x.shape[1], m.X.shape[0]

        #compute the likelihood of each batch sample (equation at the end of 2.2 in the draft)
        val_s = self.opt_val_s(f,y,m,sqrt_nu)

        #compute the gradient of the likelihood of each batch sample using generalized chain rule
        val_s_grad = self.opt_val_s_grad(s, f, y, m, sqrt_nu, L, L_inv, w_vec, w_inv, x)

        grad = np.zeros((b, d))
        for i in range(b): #Loop through winner index and compute the gradient of the likelihood
            grad = grad +  np.mean(2.0*val_s[:,i,None, None] * val_s_grad[:,i,:,:], axis=0) - 2.0 * np.mean(val_s[:,i] ,axis=0) * np.mean(val_s_grad[:,i,:,:] ,axis=0)
        return grad

    def opt_val_s(self, f: np.ndarray, y: np.ndarray, m: ComparisonGP, sqrt_nu: np.ndarray) -> np.ndarray:
        """
        Computes $ \prod_{i=1, i \neq j}^q p(y_j < y_i)$ for each sample from the latent function.
        
        :param f: latent function samples of the posterior
        :param y: observations from the latent functio (latent function values corrupted with noise)
        :param m: the model which posterior is used
        :param sqrt_nu: noise std for the observations
        :return: product of probabilities of comparisons for each sample
        """
        opt_vals = np.zeros(f.shape)
        for i in range(f.shape[1]):
            mi = [k for k in range(f.shape[1]) if k is not i]
            opt_vals[:,i] = np.prod(phi((f[:, mi] - y[:,i, None])/sqrt_nu[None,mi]), axis=1)
        return opt_vals

    def opt_val_s_grad(self, s: np.ndarray, f: np.ndarray, y: np.ndarray, m: ComparisonGP, sqrt_nu: np.ndarray, L: np.ndarray, L_inv: np.ndarray, w_vec: np.ndarray, w_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of $\prod_{i=1, i \neq j}^q p(y_j < y_i)$ for each sample from the latent function.
        
        :param f: latent function samples of the posterior
        :param y: observations from the latent functio (latent function values corrupted with noise)
        :param m: the model which posterior is used
        :param sqrt_nu: noise std for the observations
        :return: gradient of product of probabilities of comparisons for each sample
        """
        N, b, d, n = f.shape[0], x.shape[0], x.shape[1], m.X.shape[0]
        #Compute gradients using the generalized chain rule:
        opt_val_s_gradient_f = self.dopt_val_s_df(f, y,m, sqrt_nu) # N x b x b
        gradient_f_x = self.dgp_dx(s, L, L_inv, w_vec, w_inv, m.kern, x, m.X) # N x b x b x d
        grad = np.zeros((N, b , b, d))
        for k in range(b): #batch winner
            for i in range(b): #batch member
                for j in range(d): # dimension of the batch member
                    grad[:,k,i,j] = np.matmul(opt_val_s_gradient_f[:,k,None,:], gradient_f_x[:,:,None,i,j]).flatten()
        return grad

    def dopt_val_s_df(self, f: np.ndarray, y: np.ndarray, m: ComparisonGP, sqrt_nu: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of $\prod_{i=1, i \neq j}^q p(y_j < y_i)$ for each sample from the latent function with respect to the GP
        
        :param f: latent function samples of the posterior
        :param y: observations from the latent functio (latent function values corrupted with noise)
        :param m: the model which posterior is used
        :param sqrt_nu: noise std for the observations
        :return: gradient of product of probabilities of comparisons for each sample with respect to the GP
        """
        dl = np.zeros((f.shape[0], f.shape[1], f.shape[1])) # samples, winner , derivative index
        for i in range(f.shape[1]): #loop through winner index
            for j in range(f.shape[1]): #loop through f to derivate with respect to
                if j is not i:
                    mij = [k for k in range(f.shape[1]) if (k is not i) and (k is not j)]
                    dl[:,i,i] += -1.0/sqrt_nu[j]*npdf((f[:,j]-y[:,i])/sqrt_nu[j],0,1)*np.prod(phi((f[:, mij] - y[:, i, None])/sqrt_nu[None,mij]), axis=1)
                    dl[:,i,j] = 1.0/sqrt_nu[j]*npdf((f[:,j]-y[:,i])/sqrt_nu[j],0,1)*np.prod(phi((f[:, mij] - y[:, i, None])/sqrt_nu[None,mij]), axis=1)
        return dl # n x b x b
    
class QExpectedImprovement(ExpectationAcquisition):
    """
    This class implements the q Expected Improvement acquisition function
    """
    def opt_val(self, f: np.ndarray ,y: np.ndarray, m: ComparisonGP, sqrt_nu: np.ndarray) -> float:
        """
        computes the acquisition function value by sampling
        
        :param f: latent function samples of the posterior
        :param y: observations from the latent functio (latent function values corrupted with noise)
        :param m: the model which posterior is used
        :param sqrt_nu: noise std for the observations
        :return: q Expected Improvement at the input
        """
        yw = m.get_current_best()
        liks = np.zeros(f.shape)
        for i in range(f.shape[1]): #Loop through batch winners (lower value better)
            mi = [k for k in range(f.shape[1]) if k is not i]
            liks[:,i] = (y[:,i]<yw)*np.array(yw-y[:,i])*np.prod(phi((f[:, mi] - y[:,i, None])/sqrt_nu[None,mi]), axis=1)
        return np.mean(np.sum(liks, axis=1), axis=0)

    def opt_val_grad(self, s: np.ndarray, f: np.ndarray, y: np.ndarray, m: ComparisonGP,
                     sqrt_nu: np.ndarray, L: np.ndarray, L_inv: np.ndarray, w_vec: np.ndarray,
                     w_inv: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        computes the acquisition function value by sampling
        
        :param f: latent function samples of the posterior
        :param y: observations from the latent functio (latent function values corrupted with noise)
        :param m: the model which posterior is used
        :param sqrt_nu: noise std for the observations
        :param L: Cholesky decomposition(s) of the posterior covariance matrix (samples)
        :param L_inv: Inverse(s) of Cholesky decomposition(s) of the posterior covariance matrix (samples)
        :param w_vec: woodbury vectors of the posterior of the model
        :param w_inv: inverses of the woodbury matrix of the model
        :param x: The locations the samples are taken at      
        :return: Gradient of q Expected Improvement at the input
        """
        N, b, d, n = f.shape[0], x.shape[0], x.shape[1], m.X.shape[0]
        #Compute gradients using the generalized chain rule:
        opt_val_gradient_f = self.dopt_val_df(f, y,m, sqrt_nu) # N x b
        gradient_f_x = self.dgp_dx(s, L, L_inv, w_vec, w_inv, m.kern, x, m.X) # N x b x b x d
        grad = np.zeros((N, b, d))
        for i in range(b):
            for j in range(d):
                grad[:,i,j] = np.matmul(opt_val_gradient_f[:,None,:], gradient_f_x[:,:,None,i,j]).flatten()
        return np.mean(grad, axis=0)

    def dopt_val_df(self, f: np.ndarray, y: np.ndarray, m: ComparisonGP, sqrt_nu: np.ndarray) -> np.ndarray:
        """
        computes the gradient of the acquisition function value with respect to each input location and gp sample
        
        :param f: latent function samples of the posterior
        :param y: observations from the latent functio (latent function values corrupted with noise)
        :param m: the model which posterior is used
        :param sqrt_nu: noise std for the observations     
        :return: gradient of the acquisition function value with respect to each input location and gp sample
        """
        yw = m.get_current_best()
        dl = np.zeros(f.shape)
        for i in range(f.shape[1]): #loop through f to derivate with respect to
            mi = [k for k in range(f.shape[1]) if (k is not i)]
            dl[:,i] = dl[:,i] - np.prod(phi((f[:, mi] - y[:,i, None])/sqrt_nu[None,mi]), axis=1)*(y[:,i]<yw)
            for j in range(f.shape[1]):
                if j is not i:
                    mij = [k for k in range(f.shape[1]) if (k is not i) and (k is not j)]
                    dl[:,i] = dl[:,i] \
                        - (y[:,i]<yw)*(yw - y[:,i])/sqrt_nu[j]*npdf((f[:,j]-y[:,i])/sqrt_nu[j],0,1)*np.prod(phi((f[:, mij] - y[:, i, None])/sqrt_nu[None,mij]), axis=1) \
                        + (y[:,j]<yw)*(yw - y[:,j])/sqrt_nu[i]*npdf((f[:,i]-y[:,j])/sqrt_nu[i],0,1)*np.prod(phi((f[:, mij] - y[:, j, None])/sqrt_nu[None,mij]), axis=1)   # first line: i is winner second line: j is winner
        return dl # n x b
