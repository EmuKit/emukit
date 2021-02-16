
import numpy as np
import scipy as sp
from scipy.integrate import quad, dblquad

from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp
from scipy.linalg import sqrtm, inv

import GPy


from GPy.util.linalg import dtrtrs, dpotrs, pdinv, tdot, jitchol
from GPy.util.univariate_Gaussian import std_norm_pdf, std_norm_cdf, derivLogCdfNormal, logCdfNormal, cdfNormal
from GPy.inference.latent_function_inference.posterior import Posterior
from scipy.linalg import block_diag
from GPy.util import choleskies

from typing import Callable, List, Tuple, Dict

from ..util import adam

#Helper functions
phi = lambda x: cdfNormal(x)
sigmoid = lambda x: 1./(1 + np.exp(-x))

dlogphi_df = lambda x: derivLogCdfNormal(x)
dlogsigmoid_df = lambda x: sigmoid(-x)
d2logsigmoid_df = lambda x: -sigmoid(-x)*sigmoid(x)

def dL_fr(L: np.ndarray, dsigma: np.ndarray, alpha: np.ndarray, beta: np.ndarray, K: np.ndarray):
    """
    Partial derivative of function with respect to beta using cholesky decomposition and the generalized matrix chain rule
    The ending _fr comes from "full rank", this method should be used when we use the full rank parametrization
    
    :param L: Cholesky decomposition of Sigma: Sigma= L L^T
    :param dsigma: derivative of the function with respect to Sigma
    :param alpha: alpha parameter
    :param beta: beta parameter, the vector the derivative is taken with respect to
    :param K: prior covariance matrix
    :return: The derivative of function with respect to beta
    """
    Sigma = L @ L.T
    t2 = np.zeros_like(dsigma)
    for m in range(t2.shape[0]):
        for n in range(m+1):
            dl = np.zeros_like(dsigma)
            dl[m,:] = L[:,n].T
            dl += dl.T
            t2[m,n] =  np.trace(dsigma.T @ dl)
    return t2[np.newaxis,:]

def dL_mf(L: np.ndarray, dsigma: np.ndarray, alpha: np.ndarray, beta: np.ndarray, K: np.ndarray):
    """
    Partial derivative of function with respect to beta using cholesky decomposition and the generalized matrix chain rule
    The ending _fr comes from "mean field", this method should be used when we use the meaan field parametrization
    
    :param L: Cholesky decomposition of Sigma: Sigma= L L^T
    :param dsigma: derivative of the function with respect to Sigma
    :param alpha: alpha parameter
    :param beta: beta parameter, the vector the derivative is taken with respect to
    :param K: prior covariance matrix
    :return: The derivative of function with respect to beta
    """
    dL = np.zeros((L.shape[0], L.shape[1], beta.shape[0]))
    dL2 = np.zeros((L.shape[0], L.shape[1], beta.shape[0]))
    res = np.zeros_like(beta)

    S = L @ L.T
    for k in range(beta.shape[0]):
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                dL[i, j, k] = -2.0*beta[k]*S[i, k]*S[j, k]
        res[k] += np.trace(dsigma.T @ dL[:, :, k])
    return res

def dSigma_dLmn(L: np.ndarray, m: int, n: int):
    """
    Partial derivative of Sigma with respect to one element of L when Sigma=L L^T
    
    :param L: Cholesky decomposition of sigma
    :param m: row index in L
    :param n: column index in L
    :return: partial derivative of Sigma
    """
    delta = 1e-5
    L_new = L.copy()
    L_new[m,n] += delta
    Sigma_new = L_new @ L_new.T
    Sigma = L @ L.T
    return (Sigma_new-Sigma)/delta

def comp_y_ij(mu: np.ndarray, Sigma: np.ndarray, i: int, j: int, epsilon: float):
    """
    A helper method to compute the mean and covariance of y_j-y_j when we know the joint distribution of f and y=f+noise, where noise has standard deviation epsilon
    :param mu: mean of latent function f
    :param Sigma: covariance of latent function f
    :param i: index i in y_j - y_i
    :param j: index j in y_j - y_i
    :param epsilon: noise standard deviation of y
    """
    m_diff = mu[j] - mu[i]
    sigma_diff =  np.sqrt(Sigma[i,i] + Sigma[j,j] + 2*Sigma[i,j])
    return m_diff + sigma_diff*epsilon, sigma_diff


def variational_expectations_ove_full_rank(mf: np.ndarray, Sigma: np.ndarray, ind_winners: List[int], ind_loosers: List[int], sigma2s: float=1.) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes the variational expectation and derivatives for the full rank approximation for a single batch
    
    :param mf: mean of the latent function approximation of the batch
    :param Sigma: Covariance of the latent function approximation of the batch
    :param ind_winners: List of batch winners in each pairwise comparisons. We assume that the feedback is given as batch winner form
    :param ind_loosers: List of batch loosers in each pairwise comparisons
    :param sigma2s: noise variance of the observations
    :return: expectation and its derivatives with respect to mean and covariance
    """
    N = mf.shape[0]

    dF_dm = np.zeros((N,1))
    dF_dSigma = np.zeros((N,N))

    #Integration by quadrature
    gh_x, gh_w = np.polynomial.hermite.hermgauss(25)
    gh_w = gh_w / np.sqrt(np.pi)

    #to make sigmoid look more like probit
    sigma2s = sigma2s/1.6

    F = 0
    
    # i is the winner:
    i = ind_winners[0]
    for j in ind_loosers:
        y_ij, sigma_ij = comp_y_ij(mf, Sigma, i, j, gh_x)
        F += np.sum(np.log(sigmoid(y_ij/(np.sqrt(2)*sigma2s[i])))*gh_w )
        ms_y_ij = sigmoid(-y_ij/(np.sqrt(2)*sigma2s[i]) )
        dF_dm[i,0] += np.sum(-gh_w*ms_y_ij/(np.sqrt(2)*sigma2s[i]) )
        dF_dm[j,0] = np.sum(gh_w*ms_y_ij/(np.sqrt(2)*sigma2s[i]))
        dF_dSigma[j,j] = 0.5*np.sum(gh_w*ms_y_ij/sigma_ij*gh_x/(np.sqrt(2)*sigma2s[i]))
        dF_dSigma[i, i] += dF_dSigma[j,j].copy()
        dF_dSigma[i,j] = 2.0*dF_dSigma[j,j]
    return F, dF_dm, dF_dSigma


def df_d(y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]], m: np.ndarray, L: np.ndarray, L_inv: np.ndarray, K: np.ndarray, sigma2s: np.ndarray, alpha: np.ndarray, beta: np.ndarray, s_to_l: Callable=dL_fr):
    """
    Computes the log marginal likelihood and its derivatives with respect to alpha and beta. Works for both mean feald and full rank approximations 
    
    :param y: Direct observations in as a list of tuples telling location index (row in X) and observation value.
    :param yc: Batch comparisons in a list of lists of tuples. Each batch is a list and tuples tell the comparisons (winner index, loser index)
    :param m: mean of the latent values
    :param L: Cholesky decomposition of the latent value covariance
    :param L_inv: inverse of the cholesky decomposition
    :param K: prior covariance
    :param sigma2s: noise variance of the observations
    :param alpha: Alpha vector used to parametrize the posterior approximation
    :param beta: Beta vector/matrix used to parametrize the posterior approximation
    :param s_to_l: A function to compute the derivative of log likelihood with respect to beta using the generalized chain rule and when we know the derivative of log likelihood with respect to Sigma
    :return: A tuple containing log marginal likelihood, its derivative with respect to alpha and its derivative with respect to beta
    """
    Sigma = L @ L.T

    dF_dm_full = np.zeros_like(m)
    dF_dSigma_full = np.zeros_like(Sigma)
    F_full = 0
    #log_marginal = 0
    d_list = np.random.choice(range(len(yc)), size=len(yc), replace=False)
    for batch_idx in d_list:
        loc_inds_winners, loc_inds_losers = [yc[batch_idx][k][0] for k in range(len(yc[batch_idx]))], [yc[batch_idx][k][1] for k in range(len(yc[batch_idx]))]
        loc_inds_batch = np.sort(np.unique(loc_inds_winners + loc_inds_losers))
        # get winners
        ind_winners, ind_losers = [np.where(loc_inds_batch == it)[0][0] for it in loc_inds_winners], [np.where(loc_inds_batch == it)[0][0] for it in loc_inds_losers]

        # get variational moments
        F_batch, dF_dm_batch, dF_dSigma_batch = variational_expectations_ove_full_rank(m[loc_inds_batch], Sigma[np.ix_(loc_inds_batch, loc_inds_batch)], ind_winners, ind_losers, sigma2s[loc_inds_batch])
        dF_dm_full[loc_inds_batch] += dF_dm_batch
        dF_dSigma_full[np.ix_(loc_inds_batch, loc_inds_batch)] += dF_dSigma_batch
        F_full += F_batch

    #delta = 1e-5
    if len(y) > 0:
        ys = np.zeros((len(y),1))
        y_inds = np.zeros(len(y), dtype=int)
        #dir_list = np.random.choice(range(len(y)), size=len(y), replace=False)
        for ind in range(len(y)):
            (y_inds[ind], ys[ind,0]) = y[ind] #index in kernel, y value
        F_full += -0.5*np.sum(  ( (m[y_inds] - ys)**2 + Sigma[y_inds, y_inds].reshape((-1,1)) ) / sigma2s[y_inds].reshape((-1,1)) )
        dF_dm_full[y_inds] += (ys - m[y_inds] ) / sigma2s[y_inds].reshape((-1,1))
        dF_dSigma_full[y_inds, y_inds] += -0.5 / sigma2s[y_inds].reshape((-1))

    alpha_grad = K @ dF_dm_full

    beta_grad = s_to_l(L, dF_dSigma_full, alpha, beta, K)

    log_marginal = F_full
    if beta_grad.shape[1] > 1:
        beta_grad = choleskies._triang_to_flat_pure(beta_grad)
    return log_marginal, alpha_grad, beta_grad

def recompute_posterior_fr(alpha: np.ndarray, beta: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recompute the posterior approximation (for the full rank approximation) mean: K alpha, covariance inv(K + beta)
    :param alpha: Alpha vector used to parametrize the posterior approximation
    :param beta: Beta vector/matrix used to parametrize the posterior approximation
    :param K: prior covariance
    :return: Tuple containing the mean and cholesky of the covariance, its inverse and derivatives of the KL divergence with respect to beta and alpha
    """
    N = K.shape[0]
    L = choleskies._flat_to_triang_pure(beta)
    assert(L.shape[0]==1)
    L = L[0,:,:]
    lam_sqrt= np.diag(L)
    lam = lam_sqrt**2

    # Compute Mean
    m = K @ alpha
    jitter = 1e-5
    dKL_da = m.copy()
    Kinv  = np.linalg.inv(K+ np.eye(N)*jitter)
    L_inv  = np.linalg.inv(L)

    Sigma = np.empty((alpha.size, alpha.shape[0]))
    Lamda_full_rank = np.dot(L, L.T)

    dKL_db_triang = -dL_fr(L, 0.5*(np.linalg.inv(Lamda_full_rank) - Kinv), None, None, None)

    mat1 = np.linalg.inv(K + Lamda_full_rank)
    #Sigma = np.linalg.inv(Kinv + np.linalg.inv(Lamda_full_rank))
    Sigma = Lamda_full_rank
    # Compute KL
    KL = 0.5*(-N + (m.T@Kinv@m) + np.trace(Kinv @ Sigma) - np.log(np.linalg.det(Sigma @ Kinv)))
    dKL_db = choleskies._triang_to_flat_pure(dKL_db_triang)

    return m, L, L_inv, KL, dKL_db, dKL_da

def recompute_posterior_mf(alpha: np.ndarray, beta: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recompute the posterior approximation (for the mean field approximation) mean: K alpha, covariance inv(K + beta)
    :param alpha: Alpha vector used to parametrize the posterior approximation
    :param beta: Beta vector/matrix used to parametrize the posterior approximation
    :param K: prior covariance
    :return: Tuple containing the mean and cholesky of the covariance, its inverse and derivatives of the KL divergence with respect to beta and alpha
    """
    N = alpha.shape[0]
    # Lambda = diag(lam) = diag(beta.^2)
    lam_sqrt = beta.ravel()
    lam = beta.ravel() ** 2

    # Handle A = I + Lambda*K*Lambda
    KB = K @ np.diag(lam_sqrt)
    BKB = np.diag(lam_sqrt) @ KB
    A = np.eye(N) + BKB
    Ai, LA, Li, Alogdet = pdinv(A)

    # Compute Mean
    m = K @ alpha

    # Compute covariance matrix
    W = Li @ np.diag(1. / lam_sqrt)  # can be accelerated using broadcasting instead of matrix multiplication
    Sigma = np.diag(
        1. / lam) - W.T @ W  # computes np.diag(1./lam) - np.diag(1. / lam_sqrt) @ Ai @ np.diag(1. / lam_sqrt)

    # Compute KL
    KL = 0.5 * (Alogdet + np.trace(Ai) - N + np.sum(m * alpha))

    # Compute Gradients
    A_A2 = Ai - Ai.dot(Ai)
    dKL_db = np.diag(np.dot(KB.T, A_A2)).reshape(-1,1)
    # dKL_da = K @ alpha
    dKL_da = m.copy()

    L = GPy.util.linalg.jitchol(Sigma)
    L_inv = np.linalg.inv(L)

    return m, L, L_inv, KL, dKL_db, dKL_da

def log_lik(x: np.ndarray, arg_list: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Callable]) -> Tuple[np.array, np.array]:
    """
    Computes the log likelihood and gradients for specific alpha and beta values concatenated in x.
    
    :param x: Concatenated and flattened alpha and beta
    :param arg_list: List of arguments that don't change during the optimization 
                     (prior covariance, noise of the observations, observations,
                     comparisons, function to transfer the partial derivative)
    :return: Tuple containing the log marginal and its derivative with respect to alpha and beta.
    """
    
    K, sigma2s, y, yc, recompute_posterior, s_to_l = arg_list[0], arg_list[1], arg_list[2], arg_list[3], arg_list[4], arg_list[5]
    alpha = x[:K.shape[0]].reshape(-1,1)
    beta = x[K.shape[0]:].reshape(-1,1)

    if not isinstance(sigma2s, np.ndarray):
        sigma2s = sigma2s*np.ones((K.shape[0], 1))

    m, L, L_inv, KL, dKL_db, dKL_da = recompute_posterior(alpha, beta, K)

    log_marginal, alpha_grad, beta_grad = df_d(y, yc, m, L, L_inv, K, sigma2s, alpha, beta, s_to_l=s_to_l)

    log_marginal -= KL.sum()
    alpha_grad -=  dKL_da
    beta_grad -= dKL_db
    return -log_marginal, -np.r_[alpha_grad, beta_grad].reshape(-1)

def vi_comparison(X: np.ndarray, y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]],
                  kern: GPy.kern.Kern, sigma2s: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
                  max_iters: int=200, lr: float=1e-3, method: str='fr', optimize: str="adam",
                  get_logger: Callable=None) -> Tuple[Posterior, float, Dict, np.ndarray, np.ndarray]:
    """
    :param X: All locations of both direct observations and batch comparisons
    :param y: Direct observations in as a list of tuples telling location index (row in X) and observation value.
    :param yc: Batch comparisons in a list of lists of tuples. Each batch is a list and tuples tell the comparisons (winner index, loser index)
    :param kern: Prior covariance kernel
    :param sigma2s: Noise variance of the observations
    :param alpha: Initial values for alpha
    :param beta: Initial values for beta
    :param max_iter: macimum number of optimization iterations
    :param method: full rank 'fr' or mean field 'mf' methods
    :param optimize: optimization algorithm. adam or l-bfgs-B
    :param get_logger: Function for receiving the legger where the prints are forwarded.
    :return: A Tuple containing the posterior, log marginal likelihood, its gradients with respect to hyper parameters (not supported at the moment) and alpha and beta values
    """
    if(method == 'fr'):
        recompute_posterior = recompute_posterior_fr
        s_to_l = dL_fr
    else:
        recompute_posterior = recompute_posterior_mf
        s_to_l = dL_mf

    K = kern.K(X)
    K = K + 1e-6*np.identity(len(K))
    N = X.shape[0]

    X0 = np.r_[alpha, beta]
    args = [K, sigma2s, y, yc, recompute_posterior, s_to_l]
    if optimize == "adam":
        X, log_marginal, _ = adam(log_lik, X0.flatten(), args, bounds=None, max_it=max_iters, get_logger=get_logger)
    else:
        res = sp.optimize.minimize(fun=log_lik,
                                   x0=X0.flatten(),
                                   args= args,
                                   method='L-BFGS-B',
                                   jac=True,
                                   bounds=None
                                   )
        X = res.x.reshape(-1)
        log_marginal = res.fun
    alpha = X[:K.shape[0]].reshape(-1,1)
    beta = X[K.shape[0]:].reshape(-1,1)

    # Create posterior instance
    m, L, L_inv, KL, dKL_db_, dKL_da_ = recompute_posterior(alpha, beta, K)
    posterior = Posterior(mean=m, cov=L @ L.T, K=K)
    grad_dict = {}# {'dL_dK': dF_dK - dKL_dK, 'dL_dthetaL':dL_dthetaL}
    # return posterior, log_marginal, grad_dict
    return posterior, log_marginal, grad_dict, alpha, beta
