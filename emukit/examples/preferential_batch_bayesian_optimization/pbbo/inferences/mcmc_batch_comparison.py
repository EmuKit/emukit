import numpy as np
import scipy.linalg as la
import os

import matplotlib.pyplot as plt


import itertools

import GPy

from copy import deepcopy

from importlib import reload

from GPy.inference.latent_function_inference.posterior import Posterior
from GPy.util.linalg import tdot, dpotrs, pdinv, jitchol, dpotri
import stan_utility
import os
import sys

from typing import Tuple, List, Callable, Dict

# IF RUNNING ON CLUSTER AND RESTRICTED TO USE ONLY ONE CORE, UNCOMMENT THIS
# set environmental variable STAN_NUM_THREADS
# Use 4 cores per chain
# os.environ['STAN_NUM_THREADS'] = "1" # To make our lives easier on clusters


def remove_level_uncertainty(samples: np.ndarray, mean_same: bool=True) -> np.ndarray:
    """
    Removes level uncertainty in the samples. If only comparison observations are used, exact latent value doesn't matter.
    Thus in this case we make all samples have same maximum/mean
    :param samples: posterior samples
    :param mean_same: if True, all samples have same mean. If False, they have same maximum
    :return prosessed_samples: samples without level uncertainty
    """
    maxes = samples.max(axis=1)
    if(mean_same):
        maxes = samples.mean(axis=1)
    reference = np.median(maxes)
    references = maxes - reference
    samples = samples - np.tile(references.reshape((-1,1)), (1,samples.shape[1]))
    return samples

class StanPosterior(Posterior):
    """
    Posterior generated from the posterior samples
    :param samples: Stan samples
    :param X: input locations
    :param kern: Kernel to produce prior covariance matrices
    :param remove_levels: Boolean indicating if the level uncertainty should be removed from the samples
    :param y: Direct observations in as a list of tuples telling location index (row in X) and observation value.
    :param yc: Batch comparisons in a list of lists of tuples. Each batch is a list and tuples tell the comparisons (winner index, loser index)
    :param Y: kernel indices of each observation
    :param get_logger: Function for receiving the legger where the prints are forwarded.
    """
    def __init__(self, samples: Dict, X: np.ndarray, kern: GPy.kern.Kern, noise: float, remove_levels: bool=True,
                 y: List[Tuple[int, float]]=None, yc: List[List[Tuple[int, int]]]=None, get_logger: Callable=None,
                 Y: np.ndarray=None):
        np.set_printoptions(precision=3, linewidth=500)
        self._K_chol, self._precision, self._woodbury_inv, self._woodbury_chol, self._woodbury_vector = None, None, None, None, None
        samples['f'] = np.atleast_2d(samples['f'])
        if remove_levels:
            if get_logger is not None:
                get_logger().info("Removed level uncertainty")
            samples['f'] = remove_level_uncertainty(samples['f'])
        self._mean = np.array(np.mean(samples['f'].T,axis=1)).reshape((-1,1))

        self._covariance = np.cov(samples['f'].T)

        self._prior_mean = 0
        
        self.samples = samples

        self.X = np.atleast_2d(X)
        
        self.Y = Y
        self.noise = noise
        self.kern = kern
        self._K = self.kern.K(self.X)

    def _get_mu_L(self, x_pred: np.ndarray, N: int=None, woodbury_inv: bool=False, with_index: int=None) -> Tuple:
        """
        Returns posterior mean and cholesky decomposition of the posterior samples
        
        :param x_pred: locations where the mean and posterior covariance are computed
        :param N: number of posterior samples
        :param woodbury_inv: boolean indicating whether the function should return woodbury_inv vector as well
        :param with_index: index of the specific posterior sample the function should return
        :return params: tuple containing the posterior means and choleskies of the covariances. Also woodbury inverses and woodbury choleskies if woodbury_inv is true
        """
        indices = np.arange(self.samples['f'].shape[0])
        if N is not None:
            indices = np.random.choice(indices, N)
        if with_index is not None:
            indices = np.array([with_index], dtype=int)
        N = len(indices)
        x_pred = np.atleast_2d(x_pred)
        f2_mu = np.empty((N,x_pred.shape[0]))
        f2_L = np.empty((N,x_pred.shape[0], x_pred.shape[0]))
        k_x1_x2 = self.kern.K(self.X, x_pred)
        k_x2_x2 = self.kern.K(x_pred)
        for ni, i in enumerate(indices):
            L_div_k_x1_x2 = la.solve_triangular(self.samples['L_K'][i,:,:], k_x1_x2, lower=True, overwrite_b=False)
            f2_mu[ni,:] = np.dot(L_div_k_x1_x2.T, self.samples['eta'][i,:]) #self.L_div_f[i,:])
            f2_cov = k_x2_x2 - np.dot(L_div_k_x1_x2.T, L_div_k_x1_x2)
            f2_L[ni,:,:] = jitchol(f2_cov)
        if woodbury_inv:
            w_inv = np.empty((N,self.X.shape[0],self.X.shape[0]))
            w_chol = np.empty((N,self.X.shape[0],))
            for ni, i in enumerate(indices):
                L_Kinv = la.inv(self.samples['L_K'][i,:,:])
                w_inv[ni,:,:] = L_Kinv.T @ L_Kinv
                w_chol[ni,:] = (L_Kinv.T @ self.samples['eta'][i,:, None])[:, 0] # (Kinv @ self.samples['eta'][i,:, None])[:, 0] # (L_Kinv.T @ self.samples['eta'][i,:, None])[:, 0]  # self.L_div_f[i,:]
            return f2_mu, f2_L, w_inv, w_chol
        else:
            return f2_mu, f2_L

    def _raw_predict_raw(self, x_pred: np.ndarray, include_likelihood: bool=False, samples: int=2000) -> np.ndarray:
        """
        Draw posterior predictive samples from the posterior distribution
        
        :param x_pred: locations at which the samples are wanted
        :param include_likelihood: boolean indicating if observation noise should be added to the samples
        :param samples: number of posterior predictive samples
        :return samples: posterior predictive samples
        """
        x_pred = np.atleast_2d(x_pred)
        f2_mu, f2_L = self._get_mu_L(x_pred, N=samples)
        f2 = f2_mu + np.matmul(f2_L, np.random.normal(size=(samples, x_pred.shape[0], 1)))[:,:,0]
        if include_likelihood:
            f2 = f2 + self.noise*np.random.normal(size=(samples, x_pred.shape[0]))
        return f2