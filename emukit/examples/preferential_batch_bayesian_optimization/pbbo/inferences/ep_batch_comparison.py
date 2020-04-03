import numpy as np
import time
from scipy.integrate import quad, dblquad
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp
from scipy.linalg import sqrtm, inv, svd
import scipy.linalg as la

import GPy
import nearestPD

from GPy.inference.latent_function_inference.expectation_propagation import posteriorParams #, gaussianApproximation
from GPy.inference.latent_function_inference.posterior import PosteriorEP as Posterior

from GPy.util.linalg import  dtrtrs, dpotrs, tdot, symmetrify, jitchol, pdinv

from typing import Tuple, List, Callable, Dict

from .. import util


# Some helper functions: 
npdf = lambda x, m, v: 1./np.sqrt(2*np.pi*v)*np.exp(-(x-m)**2/(2*v))
log_npdf = lambda x, m, v: -0.5*np.log(2*np.pi*v) -(x-m)**2/(2*v)
phi = lambda x: norm.cdf(x)
logphi = lambda x: norm.logcdf(x)

log_2_pi = np.log(2*np.pi)

def posdef_sqrtm(M: np.ndarray) -> np.ndarray:
    """
    Returns a square root of a positive definite matrix
    :param M: A positive definite matrix
    :return: Squarte root of M
    """
    (U,S,VT) = svd(M)
    D = np.diag(np.sqrt(S))
    return U @ D @ VT

def sqrtm_block(M: np.ndarray, y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]]) -> np.ndarray:
    """
    Returns a square root of a positive definite matrix
    :param M: A positive definite block matrix
    :param y: Observations indicating where we have a diagonal element
    :param yc: Comparisons indicating where we have a block diagonal element
    :return: Squarte root of M
    """
    Msqrtm = np.zeros(M.shape)
    if(len(y)>0):
        for yi, yval in y:
            Msqrtm[yi,yi] = np.sqrt(M[yi,yi])
    if(len(yc)>0):
        for ycb in yc:
            loc_inds_winners, loc_inds_loosers = [ycb[k][0] for k in range(len(ycb))], [ycb[k][1] for k in range(len(ycb))]
            batch = np.sort(np.unique(loc_inds_winners + loc_inds_loosers))
            Msqrtm[np.ix_(batch,batch)] = posdef_sqrtm(M[np.ix_(batch,batch)])
    return Msqrtm

class MarginalMoments(object):
    """
    A simple container for the marginal moments
    :param num_data: how many observations are there
    """
    def __init__(self, num_data):
        self.logZ_hat = np.empty(num_data,dtype=np.float64)
        self.mu_hat = np.empty(num_data,dtype=np.float64)
        self.sigma2_hat = np.empty((num_data,num_data),dtype=np.float64)

class GaussianApproximation(object):
    """
    A simple container for Gaussian approximations of the batches
    :param v: scale
    :param tau: precision
    """
    def __init__(self, v, tau):
        self.v = v
        self.tau = tau

    def _update_batch(self, eta: float, delta: float, post_params: posteriorParams, marg_moments: MarginalMoments, batch: List[int], get_logger: Callable=None, sigma2s: np.ndarray=None):
        """
        Computes new gaussian approximation for a batch given posterior and marginal moments. See e.g. 3.59 in http://www.gaussianprocess.org/gpml/chapters/RW.pdf
        :param eta: parameter for fractional updates.
        :param delta: damping updates factor.
        :param post_params: Posterior approximation
        :param marg_moments: Marginal moments at this iteration
        :param batch: list of indices of the parameters to be updated
        :param get_logger: Function for receiving the legger where the prints are forwarded.
        """
        sigma_hat_inv,_,_,_ = pdinv(marg_moments.sigma2_hat[np.ix_(batch,batch)])
        post_sigma_inv,_,_,_ = pdinv(post_params.Sigma[np.ix_(batch,batch)])

        tmp0 = sigma_hat_inv - post_sigma_inv

        delta_tau = delta/eta* tmp0
        delta_v = delta/eta*(np.dot(marg_moments.mu_hat[batch],sigma_hat_inv) - np.dot(post_params.mu[batch], post_sigma_inv))
        tau_tilde_prev = self.tau[np.ix_(batch,batch)]
        tmp = (1-delta)*self.tau[np.ix_(batch,batch)] + delta_tau
        
        #Let us umake sure that sigma_hat_inv-post_sigma_inv is positive definite        
        tmp, added_value = nearestPD.nearest_pd.nearestPD(tmp)        
        update = True        
        if (added_value > 1) and (sigma2s is not None):
            update = False                
            sigma2s *= 1.05
            if get_logger is not None:
                get_logger().error('Increasing batch noise. Not updating gaussian approximation ({})'.format(sigma2s[0]))
        if update:        
            self.tau[np.ix_(batch,batch)] = tmp 
            self.v[batch] = (1-delta)*self.v[batch] + delta_v
        return (delta_tau, delta_v), sigma2s

class CavityParams(object):
    """
    A simple container for the cavity parameters
    :param num_data: how many observations are there
    """
    def __init__(self, num_data: int):
        self.v = np.zeros(num_data,dtype=np.float64)
        self.tau = np.zeros((num_data,num_data),dtype=np.float64)

    def _update_i(self, eta: float, ga_approx: GaussianApproximation, post_params: posteriorParams, i: int):
        """
        Computes the cavity params for specific index
        :param eta: parameter for fractional updates.
        :param ga_approx: Gaussian approximation
        :param post_params: Posterior approximation
        :param i: index of the parameters to be updated
        """
        self.tau[i,i] = 1./post_params.Sigma[i,i] - eta*ga_approx.tau[i,i]
        self.v[i] = post_params.mu[i]/post_params.Sigma[i,i] - eta*ga_approx.v[i]

    def _update_batch(self, eta: float, ga_approx: GaussianApproximation, post_params: posteriorParams, batch: List[int], get_logger: Callable=None):
        """
        Computes the cavity params for specific batch
        :param eta: parameter for fractional updates.
        :param ga_approx: Gaussian approximation
        :param post_params: Posterior approximation
        :param batch: list of indices of the parameters to be updated
        :param get_logger: Function for receiving the legger where the prints are forwarded.
        """
        post_sigma_inv = inv(post_params.Sigma[np.ix_(batch,batch)])
        tmp = post_sigma_inv - eta*ga_approx.tau[np.ix_(batch,batch)]
        self.tau[np.ix_(batch,batch)], _ = nearestPD.nearest_pd.nearestPD(tmp)
        self.v[batch] = np.dot(post_sigma_inv, post_params.mu[batch]) - eta*ga_approx.v[batch]

def update_posterior(K: np.ndarray, v: np.ndarray, tau: np.ndarray, y: List[Tuple[int, float]],
                     yc: List[List[Tuple[int, int]]], jitter: float=1e-9, get_logger: Callable=None) -> posteriorParams:
    """
    Update the posterior approximation. See e.g. 3.59 in http://www.gaussianprocess.org/gpml/chapters/RW.pdf
    :param K: prior covariance matrix
    :param v: Scale of the Gaussian approximation
    :param tau: Precision of the Gaussian approximation
    :param y: Observations indicating where we have a diagonal element
    :param yc: Comparisons indicating where we have a block diagonal element
    :param jitter: small number added to the diagonal to increase robustness.
    :param get_logger: Function for receiving the legger where the prints are forwarded.
    :return: posterior approximation
    """
    D = K.shape[0]
    sqrt_tau = sqrtm_block(tau + np.diag(jitter*np.ones((D))), y, yc)   
    G = np.dot(sqrt_tau, K)
    B = np.identity(D) + np.dot(G,sqrt_tau)    
    L = jitchol(B)
    V = np.linalg.solve(L, G)
    Sigma_full = K - np.dot(V.T, V)
    mu = np.dot(Sigma_full, v)
    Sigma = np.diag(Sigma_full)

    return posteriorParams(mu=mu, Sigma=Sigma_full, L=L)

def ep_comparison(X: np.ndarray, y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]],
                  kern: GPy.kern.Kern, sigma2s: np.ndarray, max_itt: int=50, delta: float=0.9,
                  eta: float = 1.0, tol: float=1e-6, ga_approx_old: GaussianApproximation=None,
                  get_logger: Callable=None) -> Tuple[Posterior, int, Dict, GaussianApproximation]:
    """
    :param X: All locations of both direct observations and batch comparisons
    :param y: Direct observations as a list of tuples telling location index (row in X) and observation value.
    :param yc: Batch comparisons in a list of lists of tuples. Each batch is a list and tuples tell the comparisons (winner index, loser index)
    :param kern: GP kernel
    :param sigma2s: noise variance of observations
    :param max_itt: maximum number of iterations
    :param delta: damping updates factor.
    :param eta: parameter for fractional updates.
    :param tol: tolerance after which the EP is stopped unless too many iterations have passed
    :param ga_approx_old: If there has been previous gaussian approximation, it should be passed
    :param get_logger: Function for receiving the legger where the prints are forwarded.
    :return: A tuple consisting of the posterior approximation, log marginal likelihood, radient dictionary and gaussian approximations of the batches
    """

    t0 = time.time()    

    N = X.shape[0]
    Ndir = len(y)
    Ncomp = len(yc)
    ###################################################################################
    # Contruct observations and kernels
    ###################################################################################
    K = kern.K(X)

    ###################################################################################
    # Prepare marginal moments, site approximation and cavity containers
    ###################################################################################


    f_marg_moments = MarginalMoments(N)
    f_ga_approx = GaussianApproximation(np.zeros(N,dtype=np.float64), np.zeros((N,N),dtype=np.float64))
    f_cavity = CavityParams(N)

    # insert likelihood information to each gaussian approximation
    for i in range(Ndir):
        (ii,yi) = y[i] #index in kernel, y value
        f_ga_approx.v[ii] = yi / sigma2s[i]
        f_ga_approx.tau[ii,ii] = 1./sigma2s[i]

    if ga_approx_old is not None: #If there exists old gaussian approximation, we reuse it
        N_old = ga_approx_old.tau.shape[0]
        if N-N_old > -1:
            f_ga_approx.v[:N_old] = ga_approx_old.v
            f_ga_approx.tau[np.ix_(np.arange(N_old), np.arange(N_old))] = ga_approx_old.tau


    ###################################################################################
    # Prepare global approximations
    ###################################################################################
    f_post_params = update_posterior(K, f_ga_approx.v, f_ga_approx.tau, y, yc)
    if np.any(np.isnan(f_post_params.mu)):
        if get_logger is not None:
            get_logger().error('Posterior mean contains nan in the EP approximation')


    ###################################################################################
    # Iterate
    ###################################################################################
    for itt in range(max_itt):
        old_params = np.hstack((f_post_params.mu.copy(), f_post_params.Sigma_diag.copy()))

        if get_logger is not None:
            get_logger().info('Iteration %d' % (itt + 1))
        d_list = []
        if(len(yc)>0):
            d_list = np.random.choice(range(len(yc)), size=len(yc), replace=False)
        for d in d_list: #iterate through batches
            loc_inds_winners, loc_inds_loosers = [yc[d][k][0] for k in range(len(yc[d]))], [yc[d][k][1] for k in range(len(yc[d]))]
            loc_inds_batch = np.sort(np.unique(loc_inds_winners + loc_inds_loosers))
            # get relevant EP parameters for comparison points
            f_cavity._update_batch(eta=eta, ga_approx=f_ga_approx, post_params=f_post_params, batch=loc_inds_batch, get_logger=get_logger)

            try:
                #get cavity parameters of the batch
                ind_winners, ind_loosers = [np.where(loc_inds_batch == it)[0][0] for it in loc_inds_winners], [np.where(loc_inds_batch == it)[0][0] for it in loc_inds_loosers] # indices within a batch
                f_marg_moments.logZ_hat[loc_inds_batch], f_marg_moments.mu_hat[loc_inds_batch], f_marg_moments.sigma2_hat[np.ix_(loc_inds_batch, loc_inds_batch)], sigma2s[loc_inds_batch] = \
                     _match_moments_batch(f_cavity.v[loc_inds_batch], f_cavity.tau[np.ix_(loc_inds_batch, loc_inds_batch)], ind_winners, ind_loosers, sigma2s[loc_inds_batch],  N=100000, get_logger=get_logger)
            except AssertionError as e:
                if get_logger is not None:
                    get_logger().error('Numerical problem with feedback %d in iteration %d. Skipping update' % (d, itt))
            _, sigma2s[loc_inds_batch] = f_ga_approx._update_batch(eta=eta, delta=delta, post_params=f_post_params, marg_moments=f_marg_moments, batch=loc_inds_batch, get_logger=get_logger, sigma2s = sigma2s[loc_inds_batch])
        f_post_params = update_posterior(K, f_ga_approx.v, f_ga_approx.tau, y, yc, get_logger=get_logger)

        if np.any(np.isnan(f_post_params.mu)) or np.any(np.isnan(f_post_params.mu)):
            if get_logger is not None:
                get_logger().error('Posterior mean contains nan in the EP approximation')

        # check for convergence
        new_params = np.hstack((f_post_params.mu.copy(), f_post_params.Sigma_diag.copy()))
        converged = True
        if ( np.mean((new_params-old_params)**2)/np.mean(old_params**2) < tol):
            pass
        else:
            converged = False
        if converged:
            run_time = time.time() - t0
            if get_logger is not None:
                get_logger().info('Converged in %d iterations in %4.3fs' % (itt + 1, run_time))
            break
    #############################################################################3
    # Marginal likelihood & gradients
    #############################################################################3

    # compute normalization constant for likelihoods
    for i in range(Ndir):
        (ii,yi) = y[i] #index in kernel, y value
        f_cavity._update_i(eta=eta, ga_approx=f_ga_approx, post_params=f_post_params, i=ii)
        f_marg_moments.logZ_hat[ii] = log_npdf(yi, f_cavity.v[ii]/f_cavity.tau[ii,ii], 1./f_cavity.tau[ii,ii] + sigma2s[i])

    # marginal likelihood and gradient contribution from each f

    Z_tilde = _log_Z_tilde(f_marg_moments, f_ga_approx, f_cavity, y, yc)
    f_posterior, f_logZ, f_grad = _inference(K, f_ga_approx, f_cavity, Z_tilde, y, yc)
    return f_posterior, f_logZ, f_grad, f_ga_approx

def _match_moments_batch(v_cav: np.ndarray, tau_cav: np.ndarray, ind_winners: List[int], ind_loosers: List[int],
                         nu2: np.ndarray, N: int=10000, get_logger: Callable=None) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Computes the moments of the given batch
    :param v_cav: Scale of the cavity distribution
    :param tau_cav: Precision of the cavity distribution
    :param ind_winners: Indices of winners in each pairwise comparison of the batch
    :param ind_loosers: Indices of loosers in each pairwise comparison of the batch
    :param nu2: Noise variance of each batch member likelihood
    :param N: Number of parameters used to compute the moments
    :param get_logger: Function for receiving the legger where the prints are forwarded.
    :return: Tuple containing the integral, mean and variance
    """
    n = tau_cav.shape[0]
    tau_cav_inv = inv(tau_cav)
    m_cav, sigma_cav = np.dot(v_cav, tau_cav_inv), tau_cav_inv
    logZ, site_m, site_m2 = _compute_moments_sampling(m_cav, sigma_cav, ind_winners,ind_loosers,nu2,N=N, get_logger=get_logger)

    min_eig0 = np.min(np.real(la.eig(site_m2)[0]))       
    if min_eig0 < 0:
        site_m2 -= min_eig0*np.eye(site_m2.shape[0])
        if get_logger is not None:
            get_logger().error('marginal moment not suitable in moment matching, adding smallest eigenvalue ({}) to the diagonal to guarantee PD'.format(-1*min_eig0)) 

    if np.isnan(logZ) or np.any(np.isnan(site_m)) or np.any(np.isnan(site_m2)):
        raise AssertionError()
    return np.array([logZ/n for i in range(n)]), site_m.reshape(n,), site_m2, nu2

def _log_Z_tilde(marg_moments: MarginalMoments, ga_approx: GaussianApproximation, cav_params: CavityParams, y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]]) -> float:
    """
    Give the log marginal likelihood of the posterior approximation
    :param marg_moments: Marginal moments of the posterior
    :param ga_approx: Gaussian approximation of the batches
    :param cav_params: Cavity parameters of the posterior
    :param y: Direct observations as a list of tuples telling location index (row in X) and observation value.
    :param yc: Batch comparisons in a list of lists of tuples. Each batch is a list and tuples tell the comparisons (winner index, loser index)
    :return: log marginal likleihood of the posterior
    """
    
    #go through direct observations:
    inds = [np.array([y[i][0]]) for i in range(len(y))] #direct observations
    for d in range(len(yc)): #iterate through batches
        loc_inds_winners, loc_inds_loosers = [yc[d][k][0] for k in range(len(yc[d]))], [yc[d][k][1] for k in range(len(yc[d]))]
        inds += [np.sort(np.unique(loc_inds_winners + loc_inds_loosers))]
    log_Z_tilde = np.sum(marg_moments.logZ_hat + 0.5*np.log(2*np.pi))
    for ind in inds:
        log_Z_tilde += (0.5*np.log(np.linalg.det(np.eye(len(ind))+np.dot(ga_approx.tau[np.ix_(ind,ind)], inv(cav_params.tau[np.ix_(ind,ind)]))))
                        -0.5*np.linalg.multi_dot([ga_approx.v[ind].T, inv(cav_params.tau[np.ix_(ind,ind)] + ga_approx.tau[np.ix_(ind,ind)]),ga_approx.v[ind]])
                        + 0.5*np.linalg.multi_dot([cav_params.v[ind].T, inv(cav_params.tau[np.ix_(ind,ind)] + ga_approx.tau[np.ix_(ind,ind)]),np.linalg.multi_dot([ga_approx.tau[np.ix_(ind,ind)], inv(cav_params.tau[np.ix_(ind,ind)]), cav_params.v[ind]]) -2.0 * ga_approx.v[ind] ])  )
    return log_Z_tilde

def _ep_marginal(K: np.ndarray, ga_approx: GaussianApproximation, Z_tilde: float, y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]]) -> Tuple[float, posteriorParams]:
    """
    Compute Gaussian log marginal and posterior
    :param K: prior covariance matrix
    :param ga_approx: Gaussian approximation of the batches
    :param Z_tilde: log marginal of the posterior
    :param y: Direct observations as a list of tuples telling location index (row in X) and observation value.
    :param yc: Batch comparisons in a list of lists of tuples. Each batch is a list and tuples tell the comparisons (winner index, loser index)
    """
    post_params = update_posterior(K, ga_approx.v, ga_approx.tau,y,yc) 

    # Gaussian log marginal excluding terms that can go to infinity due to arbitrarily small tau_tilde.
    # These terms cancel out with the terms excluded from Z_tilde
    B_logdet = np.sum(2.0*np.log(np.diag(post_params.L)))
    log_marginal =  0.5*(-len(ga_approx.v) * log_2_pi - B_logdet + np.sum(ga_approx.v * np.dot(post_params.Sigma,ga_approx.v)))
    log_marginal += Z_tilde

    return log_marginal, post_params

def _inference(K: np.ndarray, ga_approx: GaussianApproximation, cav_params: CavityParams, Z_tilde: float,
               y: List[Tuple[int, float]], yc: List[List[Tuple[int, int]]]) -> Tuple[Posterior, int, Dict]:
    """
    Compute the posterior approximation
    :param K: prior covariance matrix
    :param ga_approx: Gaussian approximation of the batches
    :param cav_params: Cavity parameters of the posterior
    :param Z_tilde: Log marginal likelihood
    :param y: Direct observations as a list of tuples telling location index (row in X) and observation value.
    :param yc: Batch comparisons in a list of lists of tuples. Each batch is a list and tuples tell the comparisons (winner index, loser index)
    :return: A tuple consisting of the posterior approximation, log marginal likelihood and gradient dictionary
    """
    
    log_marginal, post_params = _ep_marginal(K, ga_approx, Z_tilde,y,yc)
    tau_tilde_root = sqrtm_block(ga_approx.tau, y,yc)
    Sroot_tilde_K = np.dot(tau_tilde_root, K)
    aux_alpha , _ = dpotrs(post_params.L, np.dot(Sroot_tilde_K, ga_approx.v), lower=1)
    alpha = (ga_approx.v - np.dot(tau_tilde_root, aux_alpha))[:,None] #(K + Sigma^(\tilde))^(-1) /mu^(/tilde)
    LWi, _ = dtrtrs(post_params.L, tau_tilde_root, lower=1)

    Wi = np.dot(LWi.T,LWi)
    symmetrify(Wi) #(K + Sigma^(\tilde))^(-1)
    dL_dK = 0.5 * (tdot(alpha) - Wi)
    dL_dthetaL = 0
    return Posterior(woodbury_inv=np.asfortranarray(Wi), woodbury_vector=alpha, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL, 'dL_dm':alpha}

def _compute_moments_sampling(mf: np.ndarray, vf: np.ndarray, ind_winners: List[int], ind_loosers: List[int],
                              nu2: np.ndarray, N: int=10000, get_logger: Callable=None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the marginal moments of a batch using numerical integration
    :param mf: cavity mean of the batch
    :params vf: cavity covariance of the batch
    :params ind_winners: indices of comparison winners
    :params ind_loosers: indices of comparison loosers
    :param nu2: noise variance of the batch members
    :param N: number of samples used for approximating the moments
    :param get_logger: Function for receiving the logger where the prints are forwarded.
    :return: Tuple containing the integral, first and second moments
    """
    m = vf.shape[0]
    sqrt_nu = np.sqrt(nu2)

    f = np.random.multivariate_normal(mf,vf,N)
    
    # Generate samples to compute likelihood
    if( not (np.prod(ind_winners == ind_winners[0])==1 )): # batch winner case
        return

    ind_winner = ind_winners[0]
    s = f[:, ind_winner] + sqrt_nu[ind_winner]*np.random.randn(N)
    s_compare = phi((f[:, ind_loosers] -s[:,None])/sqrt_nu[None,ind_loosers,0])
    s_compare = np.exp(np.log(s_compare).sum(axis=1))
    likelihood = s_compare.copy()
    qs0_ = np.mean(likelihood)
    qs0_ = max([qs0_, 1e-9])

    # First moments:
    f_nonzero = likelihood[:,None]*f
    likelihood_sqrt = np.sqrt(likelihood)
    f_nonzero_sqrt = likelihood_sqrt[:,None]*f
    qs1_ = np.sum(f_nonzero,axis=0) / qs0_ / N 

    # Second moments
    diff_f = f_nonzero_sqrt - qs1_[None,:]*likelihood_sqrt[:,None]
    qs2_ = diff_f.T @ diff_f / qs0_ / N

    return qs0_, qs1_, qs2_