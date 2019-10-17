# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

import scipy.stats
import numpy as np

import itertools

from ...core.interfaces import IModel, IDifferentiable, IJointlyDifferentiable
from ...core.acquisition import Acquisition


class ExpectedImprovement(Acquisition):
    def __init__(self, model: Union[IModel, IDifferentiable], jitter: float = float(0))-> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """

        self.model = model
        self.jitter = jitter

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)
        mean += self.jitter

        y_minimum = np.min(self.model.Y, axis=0)
        u, pdf, cdf = get_standard_normal_pdf_cdf(y_minimum, mean, standard_deviation)
        improvement = standard_deviation * (u * cdf + pdf)

        return improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.

        :param x: locations where the evaluation with gradients is done.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        y_minimum = np.min(self.model.Y, axis=0)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)

        mean += self.jitter
        u, pdf, cdf = get_standard_normal_pdf_cdf(y_minimum, mean, standard_deviation)

        improvement = standard_deviation * (u * cdf + pdf)
        dimprovement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx

        return improvement, dimprovement_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return isinstance(self.model, IDifferentiable)


def get_standard_normal_pdf_cdf(x: np.array, mean: np.array, standard_deviation: np.array) \
        -> Tuple[np.array, np.array, np.array]:
    """
    Returns pdf and cdf of standard normal evaluated at (x - mean)/sigma

    :param x: Non-standardized input
    :param mean: Mean to normalize x with
    :param standard_deviation: Standard deviation to normalize x with
    :return: (normalized version of x, pdf of standard normal, cdf of standard normal)
    """
    u = (x - mean) / standard_deviation
    pdf = scipy.stats.norm.pdf(u)
    cdf = scipy.stats.norm.cdf(u)
    return u, pdf, cdf


class QExpectedImprovement(ExpectedImprovement):
    def __init__(self, model: Union[IModel, IDifferentiable, IJointlyDifferentiable], jitter: float = float(0),
                 fast_compute: bool = False, eps: float = 1e-3) -> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation for multiple points. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        Implementation based on paper and its source code:

        Fast computation of the multipoint ExpectedImprovement with
        applications in batch selection
        Chevalier, C. and Ginsbourger, D.
        International Conference on Learning and Intelligent Optimization.
        Springer, Berlin, Heidelberg, 2013.

        Source code: https://github.com/cran/DiceOptim

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        :param fast_compute: Whether to use faster approximative method.
        :param eps: Grid length for numerical derivative in approximative method.
        """
        super(QExpectedImprovement, self).__init__(model, jitter)
        self.fast_compute = fast_compute
        self.eps = eps

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the multi point Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        mean, variance = self.model.predict(x, full_cov=True)
        y_minimum = np.min(self.model.Y, axis=0)
        return -QExpectedImprovement.get_acquisition(mean.flatten(), variance,
                                                     y_minimum, self.fast_compute, self.eps)[0]

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.

        :param x: locations where the evaluation with gradients is done.
        """
        mean, variance = self.model.predict(x, full_cov=True)
        mean = mean.flatten()
        y_minimum = np.min(self.model.Y, axis=0)
        qei, pk, symmetric_term = QExpectedImprovement.get_acquisition(mean, variance, y_minimum,
                                                                       self.fast_compute, self.eps)

        mean_dx, variance_dx = self.model.get_joint_prediction_gradients(x)
        qei_grad = QExpectedImprovement.get_acquisition_gradient(mean, variance, mean_dx, variance_dx, y_minimum,
                                                                 pk, symmetric_term, self.fast_compute, self.eps)
        return -qei, -qei_grad

    @staticmethod
    def get_acquisition(mu: np.ndarray, Sigma: np.ndarray, y_minimum: float,
                        fast_compute: bool = False, eps: float = float(1e-5)) -> tuple:
        """
        Computes the multi point Expected Improvement. A helper function for the class.

        :param mu: Prediction mean at locations where the evaluation is done.
        :param Sigma: Prediction covariance at locations where the evaluation is done.
        :param y_minimum: The best value evaluated so far by the black box function.
        :param fast_compute: Whether to use faster approximative method.
        :param eps: Grid length for numerical derivative in approximative method.
        """
        q = mu.shape[0]
        pk = np.zeros((q,))
        first_term = np.zeros((q,))
        non_symmetric_term = np.zeros((q, q))
        symmetric_term = np.zeros((q, q))
        second_term = np.zeros((q,))
        for k in range(q):
            Sigma_k = covZk(Sigma, k)
            mu_k = mu[k] - mu
            mu_k[k] = mu[k]
            b_k = np.zeros((q,))
            b_k[k] = y_minimum

            pk[k] = scipy.stats.multivariate_normal.cdf(b_k - mu_k, np.zeros((q,)), Sigma_k)

            first_term[k] = (y_minimum - mu[k]) * pk[k]

            if fast_compute:
                second_term[k] = 1. / eps * (scipy.stats.multivariate_normal.cdf(b_k - mu_k
                                                                                 + eps * Sigma_k[:, k].flatten(),
                                                                                 np.zeros((q,)), Sigma_k) - pk[k])
            else:
                for i in range(q):
                    non_symmetric_term[k, i] = Sigma_k[i, k]
                    if(i >= k):
                        mik = mu_k[i]
                        sigma_ii_k = Sigma_k[i, i]
                        bik = b_k[i]
                        phi_ik = scipy.stats.norm.pdf(bik, loc=mik, scale=np.sqrt(sigma_ii_k))
                        cik = get_cik(b_k, mu_k, Sigma_k, i).flatten()
                        sigmaik = get_sigmaik(Sigma_k, i)
                        Phi_ik = scipy.stats.multivariate_normal.cdf(cik, np.zeros((q - 1,)), sigmaik)
                        symmetric_term[k, i] = phi_ik * Phi_ik
        if not fast_compute:
            symmetric_term = symmetric_term + symmetric_term.T
            symmetric_term[range(q), range(q)] = 0.5 * np.diag(symmetric_term)
            second_term = np.sum(symmetric_term * non_symmetric_term)
        opt_val = np.sum(first_term) + np.sum(second_term)
        return opt_val, pk, symmetric_term

    @staticmethod
    def get_acquisition_gradient(mu: np.ndarray, Sigma: np.ndarray, dmu_dx: np.ndarray,
                                 dSigma_dx: np.ndarray, y_minimum: float, pk: np.ndarray,
                                 symmetric_term: np.ndarray, fast_compute: bool = False,
                                 eps: float = float(1e-3)) -> np.ndarray:
        """
        Computes the gradient of multi point Expected Improvement. A helper function for the class.

        :param mu: Prediction mean at locations where the evaluation is done.
        :param Sigma: Prediction covariance at locations where the evaluation is done.
        :param dmu_dx: Prediction mean gradient at locations where the evaluation is done.
        :param Sigma: Prediction covariance gradient at locations where the evaluation is done.
        :param y_minimum: The best value evaluated so far by the black box function.
        :param pk: Probabilities of each random variable being smaller than the rest.
        :param symmetric_term: A helper matrix generated by the acquistion function.
        :param fast_compute: Whether to use faster approximative method.
        :param eps: Grid length for numerical derivative in approximative method.
        """
        q, d = Sigma.shape[0], dmu_dx.shape[2]
        grad = np.zeros((q, d))  # result
        b = np.zeros((q, 1))
        L = -np.diag(np.ones(q))
        Dpk = np.zeros((q, d))
        Sigk_dx = np.zeros((q, d, q, q))
        mk_dx = np.zeros((q, d, q))
        # First sum of the formula
        for k in range(q):
            bk = b.copy()
            bk[k, 0] = y_minimum  # creation of vector b^(k)
            Lk = L.copy()
            Lk[:, k] = 1.0  # linear application to transform Y to Zk
            tLk = Lk.T
            mk = Lk @ mu[:, None]  # mean of Zk (written m^(k) in the formula)
            Sigk = Lk @ Sigma @ tLk  # covariance of Zk
            Sigk = 0.5 * (Sigk + Sigk.T)  # numerical symetrization

            grad[k, :] = grad[k, :] - dmu_dx[k, k, :] * pk[k]  # q x q x d

            # compute gradient ans hessian matrix of the CDF term pk.
            gradpk = GPhi(bk - mk, b, Sigk)
            hesspk = HPhi(bk, mk, Sigk, gradient=gradpk)
            # import pdb; pdb.set_trace()
            # term A2
            for l, j in itertools.product(range(q), range(d)):
                Sigk_dx[l, j, :, :] = Lk @ dSigma_dx[:, :, l, j] @ tLk
                mk_dx[l, j, :] = Lk @ dmu_dx[:, l, j]
                Dpk[l, j] = 0.5 * np.sum(hesspk * Sigk_dx[l, j, :, :]) - gradpk[None, :] @ mk_dx[l, j, :, None]
            grad = grad + (y_minimum - mu[k]) * Dpk

            # term B
            if fast_compute:
                B = np.zeros((q, d))
                gradpk1 = GPhi(bk - mk + Sigk[:, k, None] * eps, b, Sigk)
                hesspk1 = HPhi(bk + Sigk[:, k, None] * eps, mk, Sigk, gradient=gradpk1)
                for l, j in itertools.product(range(q), range(d)):
                    f1 = (-(mk_dx[None, l, j, :] @ gradpk1[:, None])
                          + eps * (Sigk_dx[None, l, j, :, k] @ gradpk1)
                          + 0.5 * np.sum(Sigk_dx[l, j, :, :] * hesspk1))
                    f = -(mk_dx[None, l, j, :] @ gradpk[:, None]) + 0.5 * np.sum(Sigk_dx[l, j, :, :] * hesspk)
                    B[l, j] = 1.0 / eps * (f1 - f)
                grad = grad + B
            else:
                B1, B2, B3 = np.zeros((q, d)), np.zeros((q, d)), np.zeros((q, d))
                for i in range(q):
                    ineq = [n for n in range(q) if n is not i]
                    Sigk_ik = Sigk[i, k]
                    Sigk_ii = Sigk[i, i]
                    mk_i = mk[i]
                    mk_dx_i = mk_dx[:, :, i]
                    bk_i = bk[i, 0]
                    ck_pi = (bk[ineq, 0] - mk[ineq, 0] - (bk[i, 0] - mk[i, 0]) / Sigk_ii * Sigk[ineq, :][:, i])[:, None]
                    Sigk_pi = 0.5 * (Sigk[ineq, :][:, ineq]
                                     - 1.0 / Sigk_ii * Sigk[ineq, :][:, i, None] @ Sigk[ineq, :][:, i, None].T
                                     + (Sigk[ineq, :][:, ineq]
                                     - 1.0 / Sigk_ii * Sigk[ineq, :][:, i, None] @ Sigk[ineq, :][:, i, None].T).T)
                    Sigk_dx_ii = Sigk_dx[:, :, i, i]
                    Sigk_dx_ik = Sigk_dx[:, :, i, k]
                    phi_ik = np.max([scipy.stats.multivariate_normal.pdf(bk[i, 0], mk_i, Sigk_ii), 1e-11])
                    dphi_ik_dSig = ((bk_i - mk_i)**2 / (2.0 * Sigk_ii**2) - 0.5 / Sigk_ii) * phi_ik
                    dphi_ik_dm = (bk_i - mk_i) / Sigk_ii * phi_ik
                    Phi_ik = symmetric_term[k, i] / phi_ik
                    GPhi_ik = GPhi(ck_pi, np.zeros((q - 1, 1)), Sigk_pi)
                    HPhi_ik = HPhi(ck_pi, np.zeros((q - 1, 1)), Sigk_pi, GPhi_ik)
                    Sigk_mi = Sigk[ineq, i, None]
                    for l, j in itertools.product(range(q), range(d)):
                        # B1
                        B1[l, j] = B1[l, j] + Sigk_dx_ik[l, j] * phi_ik * Phi_ik
                        # B2
                        B2[l, j] = B2[l, j] + Sigk_ik * (mk_dx_i[l, j] * dphi_ik_dm
                                                         + dphi_ik_dSig * Sigk_dx_ii[l, j]) * Phi_ik
                        # B3
                        dck_pi = (-mk_dx[l, j, ineq] + (mk_dx_i[l, j] * Sigk_ii
                                                        + (bk_i - mk_i) * Sigk_dx_ii[l, j])
                                  / (Sigk_ii**2) * Sigk_mi[:, 0]
                                  - (bk[i, 0] - mk_i) / Sigk_ii * Sigk_dx[l, j, ineq, i])
                        SigtCross = Sigk_dx[l, j, ineq, i, None] @ Sigk_mi.T
                        dSigk_pi = (Sigk_dx[l, j, :, :][:, ineq][ineq, :]
                                    + Sigk_dx_ii[l, j] / (Sigk_ii**2) * (Sigk_mi @ Sigk_mi.T)
                                    - (1.0 / Sigk_ii) * (SigtCross + SigtCross.T))
                        B3[l, j] = (B3[l, j] + Sigk_ik * phi_ik * ((GPhi_ik.T @ dck_pi)
                                    + 0.5 * np.sum(HPhi_ik * dSigk_pi)))
                grad = grad + B1 + B2 + B3
        return grad


def covZk(Sigma: np.ndarray, k: int) -> np.ndarray:
    """
    Compute covariance of Y_k - Y_min given that Y_k is smaller than any other Y_i, i not k

    :param Sigma: Covariance of the multivariate gaussian.
    :param k: Index of random variable.
    """
    q = Sigma.shape[0]
    res = Sigma.copy()
    neqi = [j for j in range(q) if j is not k]  # Everything else but i
    for i in neqi:
        res[k, i] = Sigma[k, k] - Sigma[k, i]
        res[i, k] = res[k, i]
    for i in neqi:
        for j in [j for j in range(i, q) if j is not k]:
            res[i, j] = Sigma[i, j] + Sigma[k, k] - Sigma[k, i] - Sigma[k, j]
            res[j, i] = res[i, j]
    return res


def get_sigmaik(Sigma: np.ndarray, i: int) -> np.ndarray:
    """
    Covariances of variables k knowing the value of variable i.

    :param Sigma: Covariance of the multivariate Gaussian.
    :param i: Known variable.
    """
    result = np.zeros_like(Sigma)
    neqi = [j for j in range(Sigma.shape[0]) if j is not i]  # Everything else but i
    for u, v in itertools.product(neqi, neqi):
        result[u, v] = Sigma[u, v] - Sigma[u, i] * Sigma[v, i] / Sigma[i, i]
    return result[neqi, :][:, neqi]


def get_cik(b: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, i: int) -> np.ndarray:
    """
    Partial correlations of variables k knowing the value of variable i.

    :param b: Known values.
    :param mu: Mean of the multivariate Gaussian knowing i.
    :param Sigma: Covariance of the multivariate gaussian knowing i.
    :param i: known variable.
    """
    Sigmai = Sigma[i, :] / Sigma[i, i]
    result = (b - mu) - (b[i] - mu[i]) * Sigmai
    return result[np.arange(Sigma.shape[0]) != i]


def decompo(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, k: list) -> np.ndarray:
    """
    Decompose Multivariate normal to probability of some random variables being smaller than the rest and multiply this
    with the marginal probability distribution of those random variables.

    :param x: Location of decomposition.
    :param mu: Mean of the multivariate Gaussian.
    :param Sigma: Covariance of the multivariate gaussian.
    :param k: Indices of decomposition.
    """
    q = Sigma.shape[0]
    if (len(k) == q):
        return scipy.stats.multivariate_normal.pdf(x.flatten(), mu.flatten(), Sigma)
    neqk = [i for i in np.arange(q) if i not in k]
    x1 = x[k, :]
    x2 = x[neqk, :]
    mu1 = mu[k, :]
    Sig22 = Sigma[neqk, :][:, neqk]
    Sig21 = Sigma[neqk, :][:, k]
    Sig12 = Sig21.T
    Sig11 = Sigma[k, :][:, k]
    Sig11Inv = np.linalg.inv(Sig11)
    varcov = Sig22 - Sig21 @ Sig11Inv @ Sig12
    varcov = 0.5 * (varcov + varcov.T)

    if (min(np.diag(varcov)) <= 0):
        varcov[range(len(neqk)), :][:, range(len(neqk))] = np.diag(varcov) * (np.diag(varcov) >= 0) + 1e-9
    moy = mu[neqk] + Sig21 @ Sig11Inv @ (x1 - mu1)
    low = np.minimum(moy, x2) - 5.0 * np.sqrt(np.max(np.abs(varcov)))
    return scipy.stats.multivariate_normal.pdf(x1.flatten(), mu[k].flatten(), Sig11) \
        * scipy.stats.mvn.mvnun(low, x2, moy, varcov, maxpts=1000 * q)[0]


def GPhi(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Compute gradient of CDF of multivariate Gaussian distribution.

    :param x: Location where the Hessian is evaluated.
    :param mu: Mean of the multivariate Gaussian.
    :param Sigma: Covariance of the multivariate Gaussian.
    """
    return np.array([decompo(x, mu, Sigma, [i]) for i in range(Sigma.shape[0])])


def HPhi(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, gradient=None) -> np.ndarray:
    """
    Compute hessian matrix of CDF of multivariate Gaussian distribution.

    :param x: Location where the Hessian is evaluated.
    :param mu: Mean of the multivariate Gaussian.
    :param Sigma: Covariance of the multivariate Gaussian.
    :param gradient: Gradient of the multivariate Gaussian at x.
    """
    q = Sigma.shape[0]
    if (q == 1):
        res = - (x - mu) / Sigma * scipy.stats.norm.pdf(x, mu, np.sqrt(Sigma))
    else:
        res = np.zeros((q, q))
    for i in range(q - 1):
        for j in range(i + 1, q):
            res[i, j] = decompo(x, mu, Sigma, [i, j])

    res = res + res.T  # Hessian matrix is symmetric
    # diagonal terms can be computed with the gradient of CDF and the other hessian terms
    if (gradient is None):
        res = res - np.diag(((x - mu).flatten() * GPhi(x, mu, Sigma)[:, None]
                             + np.diag(Sigma @ res).flatten()) / np.diag(Sigma).flatten())
    else:
        res = res - np.diag(((x - mu).flatten() * gradient + np.diag(Sigma @ res).flatten()) / np.diag(Sigma).flatten())
    return res
