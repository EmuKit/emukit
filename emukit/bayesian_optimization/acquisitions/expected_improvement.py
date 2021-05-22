# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

import scipy.stats
import numpy as np

import itertools

from ...core.interfaces import IModel, IModelWithNoise, IDifferentiable, IJointlyDifferentiable
from ...core.acquisition import Acquisition


class ExpectedImprovement(Acquisition):
    def __init__(self, model: Union[IModel, IDifferentiable], jitter: float=0.0)-> None:
        """
        For a given input, this acquisition computes the improvement over the current best observed value in
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

        mean, variance = self._get_model_predictions(x)
        standard_deviation = np.sqrt(variance)
        mean += self.jitter

        y_minimum = self._get_y_minimum()
        u, pdf, cdf = get_standard_normal_pdf_cdf(y_minimum, mean, standard_deviation)
        improvement = standard_deviation * (u * cdf + pdf)

        return improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.

        :param x: locations where the evaluation with gradients is done.
        """

        mean, variance = self._get_model_predictions(x)
        standard_deviation = np.sqrt(variance)

        y_minimum = self._get_y_minimum()

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

    def _get_model_predictions(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions for the function values at given input locations."""
        return self.model.predict(x)

    def _get_y_minimum(self) -> np.ndarray:
        """Return the minimum value in the samples observed so far."""
        return np.min(self.model.Y, axis=0)


class MeanPluginExpectedImprovement(ExpectedImprovement):
    def __init__(self, model: IModelWithNoise, jitter: float=0.0) -> None:
        """
        A variant of expected improvement that accounts for observation noise.

        For a given input, this acquisition computes the expected improvement over the *mean* at the
        best point observed so far.

        This is a heuristic that allows Expected Improvement to deal with problems with noisy observations, where
        the standard Expected Improvement might behave undesirably if the noise is too large.

        For more information see:
            "A benchmark of kriging-based infill criteria for noisy optimization" by Picheny et al. 
        Note: the model type should be Union[IPredictsWithNoise, Intersection[IpredictsWithNoise, IDifferentiable]].
            Support for Intersection types might be added to Python in the future (see PEP 483)

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """
        super().__init__(model=model, jitter=jitter)

    def _get_y_minimum(self) -> np.ndarray:
        """Return the smallest model mean prediction at the previously observed points."""
        means_at_prev, _ = self.model.predict_noiseless(self.model.X)
        return np.min(means_at_prev, axis=0)
 
    def _get_model_predictions(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """Return the likelihood-free (i.e. without observation noise) prediction from the model."""
        return self.model.predict_noiseless(x)


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


class MultipointExpectedImprovement(ExpectedImprovement):
    def __init__(self, model: Union[IModel, IDifferentiable, IJointlyDifferentiable], jitter: float=0.0,
                 fast_compute: bool=False, eps: float=1e-3) -> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation for multiple points. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        Implementation based on papers and their implementations:

        Fast computation of the multipoint Expected Improvement with
        applications in batch selection
        Chevalier, C. and Ginsbourger, D.
        International Conference on Learning and Intelligent Optimization.
        Springer, Berlin, Heidelberg, 2013.

        Gradient of the acquisition is derived in:

        Differentiating the multipoint Expected Improvement for optimal batch design
        Marmin, S., Chevalier, C. and Ginsbourger, D.
        International Workshop on Machine Learning, Optimization and Big Data.
        Springer, Cham, 2015.

        Source code for both: https://github.com/cran/DiceOptim

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        :param fast_compute: Whether to use faster approximative method.
        :param eps: Grid length for numerical derivative in approximative method.
        """
        super(MultipointExpectedImprovement, self).__init__(model, jitter)
        self.fast_compute = fast_compute
        self.eps = eps

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the multipoint Expected Improvement.

        :param x: points where the acquisition is evaluated.
        :return: multipoint Expected Improvement at the input.
        """
        mean, variance = self.model.predict_with_full_covariance(x)
        y_minimum = np.min(self.model.Y, axis=0)
        return -self._get_acquisition(mean.flatten(), variance, y_minimum)[0]

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the multipoint Expected Improvement and its derivative.

        :param x: locations where the evaluation with gradients is done.
        :return: multipoint Expected Improvement and its gradient at the input.
        """
        mean, variance = self.model.predict_with_full_covariance(x)
        mean = mean.flatten()
        y_minimum = np.min(self.model.Y, axis=0)
        qei, pk, symmetric_term = self._get_acquisition(mean, variance, y_minimum)

        mean_dx, variance_dx = self.model.get_joint_prediction_gradients(x)
        qei_grad = self._get_acquisition_gradient(mean, variance, mean_dx, variance_dx,
                                                  y_minimum, pk, symmetric_term)
        return -qei, -qei_grad

    def _get_acquisition(self, mu: np.ndarray, Sigma: np.ndarray, y_minimum: float) -> Tuple:
        """
        Computes the multi point Expected Improvement. A helper function for the class.

        :param mu: Prediction mean at locations where the evaluation is done.
        :param Sigma: Prediction covariance at locations where the evaluation is done.
        :param y_minimum: The best value evaluated so far by the black box function.
        :return: Multipoint Expected Improvement where the mean and the covariance are evaluated.
        """
        q = mu.shape[0]
        pk = np.zeros((q,))
        first_term = np.zeros((q,))
        non_symmetric_term = np.zeros((q, q))
        symmetric_term = np.zeros((q, q))
        for k in range(q):
            Sigma_k = get_covariance_given_smallest(Sigma, k)
            mu_k = mu[k] - mu
            mu_k[k] = mu[k]
            b_k = np.zeros((q,))
            b_k[k] = y_minimum

            pk[k] = scipy.stats.multivariate_normal.cdf(b_k - mu_k, np.zeros((q,)), Sigma_k)

            first_term[k] = (y_minimum - mu[k]) * pk[k]
            symmetric_term[k, :], non_symmetric_term[k, :] = self._get_non_symmetric_and_symmetric_term_k(b_k, mu_k, Sigma_k, pk, k)

        # Symmetrify the symmetric term
        symmetric_term = symmetric_term + symmetric_term.T
        symmetric_term[range(q), range(q)] = 0.5 * np.diag(symmetric_term)
        second_term = np.sum(symmetric_term * non_symmetric_term)

        # See equation (3) in the paper for details:
        opt_val = np.sum(first_term) + np.sum(second_term)

        return opt_val, pk, symmetric_term

    def _get_acquisition_gradient(self, mu: np.ndarray, Sigma: np.ndarray, dmu_dx: np.ndarray,
                                  dSigma_dx: np.ndarray, y_minimum: float, pk: np.ndarray,
                                  symmetric_term: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of multi point Expected Improvement. A helper function for the class.

        :param mu: Prediction mean at locations where the evaluation is done.
        :param Sigma: Prediction covariance at locations where the evaluation is done.
        :param dmu_dx: Prediction mean gradient at locations where the evaluation is done.
        :param dSigma_dx: Prediction covariance gradient at locations where the evaluation is done.
        :param y_minimum: The best value evaluated so far by the black box function.
        :param pk: Probabilities of each random variable being smaller than the rest.
        :param symmetric_term: A helper matrix generated by the acquistion function.
        :return: Gradient of the multipoint Expected Improvement where the mean and the covariance are evaluated.
        """
        q, d = Sigma.shape[0], dmu_dx.shape[2]

        # Initialize empty vectors
        grad, L = np.zeros((q, d)), -np.diag(np.ones(q))

        # First sum of the formula
        for k in range(q):
            bk = np.zeros((q, 1))
            bk[k, 0] = y_minimum  # creation of vector b^(k)
            Lk = L.copy()
            Lk[:, k] = 1.0  # linear application to transform Y to Zk
            mk = Lk @ mu[:, None]  # mean of Zk (written m^(k) in the formula)
            Sigk = Lk @ Sigma @ Lk.T  # covariance of Zk
            Sigk = 0.5 * (Sigk + Sigk.T)  # numerical symmetrization

            # Compute the first two terms of the gradient (First row of Equation (6))
            grad_a, mk_dx, Sigk_dx, gradpk, hesspk = self._gradient_of_the_acquisition_first_term(mu, dmu_dx, dSigma_dx,
                                                                                                  y_minimum, pk, k, bk,
                                                                                                  mk, Sigk, Lk)
            grad = grad + grad_a

            # Compute the last three terms of the gradiens (Rows 2-4 of Equation (6))
            grad = grad + self._gradient_of_the_acquisition_second_term(mu, dmu_dx, dSigma_dx, y_minimum,
                                                                        pk, k, bk, mk, Sigk, mk_dx, Sigk_dx,
                                                                        symmetric_term, gradpk, hesspk)

        return grad

    def _get_non_symmetric_and_symmetric_term_k(self, b_k: np.ndarray, mu_k: np.ndarray,
                                                Sigma_k: np.ndarray, pk: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper function for computing Second term in Equation (3) of the paper
        for computing the multipoint Expected Improvement

        :param b_k: Vector b^{k} in Equation (3) in the paper for computing the acquisition function.
        :param mu_k: Mean of Y - Y_min given that Y_k is smaller than any other Y_i, i not k.
        :param Sigma_k: Covariance of Y - Y_min given that Y_k is smaller than any other Y_i, i not k.
        :param pk: Vector of probabilities of Y_k being smaller than the rest for all k.
        :return: Non symmetric and symmetric matrices needed for computing the multipoint Expected Improvement.
        """
        q = mu_k.shape[0]
        non_symmetric_term = np.zeros((q,))
        symmetric_term = np.zeros((q,))
        if self.fast_compute:
            non_symmetric_term[k] = 1. / self.eps * (scipy.stats.multivariate_normal.cdf(b_k - mu_k
                                                                                         + self.eps * Sigma_k[:, k].flatten(),
                                                                                         np.zeros((q,)), Sigma_k) - pk[k])
            symmetric_term[k] = 1.0

        else:
            for i in range(q):
                # First item inside the second sum of Equation (3) in the paper
                non_symmetric_term[i] = Sigma_k[i, k]

                if(i >= k):
                    mik = mu_k[i]
                    sigma_ii_k = Sigma_k[i, i]
                    bik = b_k[i]
                    phi_ik = scipy.stats.norm.pdf(bik, loc=mik, scale=np.sqrt(sigma_ii_k))
                    cik = get_correlations_given_value_of_i(b_k, mu_k, Sigma_k, i).flatten()
                    sigmaik = get_covariance_given_value_of_i(Sigma_k, i)
                    Phi_ik = scipy.stats.multivariate_normal.cdf(cik, np.zeros((q - 1,)), sigmaik)

                    # pdf times cdf in the paper, Equation (3) in the paper, two last items:
                    symmetric_term[i] = phi_ik * Phi_ik
        return symmetric_term, non_symmetric_term

    def _gradient_of_the_acquisition_first_term(self, mu: np.ndarray, dmu_dx: np.ndarray, dSigma_dx: np.ndarray,
                                                y_minimum: float, pk: np.ndarray, k: int, bk: np.ndarray,
                                                mk: np.ndarray, Sigk: np.ndarray, Lk: np.ndarray) -> Tuple[np.ndarray,
                                                np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper function for computing the first term of the gradient of the Acquisition (First row of Equation (6))

        :param mu: Prediction mean at locations where the evaluation is done.
        :param dmu_dx: Prediction mean gradient at locations where the evaluation is done.
        :param dSigma_dx: Prediction covariance gradient at locations where the evaluation is done.
        :param y_minimum: The best value evaluated so far by the black box function.
        :param pk: Probabilities of each random variable being smaller than the rest.
        :param k: Index of the assumed winner, as in Y_k < Y_i for all i not k
        :param bk: Vector b^{k} in the Equation (6) in the paper.
        :param mk: Mean of Y - Y_min given that Y_k is smaller than any other Y_i, i not k.
        :param Sigk: Covariance of Y - Y_min given that Y_k is smaller than any other Y_i, i not k.
        :param Lk: Linear application to transform Y to Zk
        :return: Gradient of the first term of the multipoint Expected Improvement,
                 gradient of the mean and covariance of the multivariate normal,
                 and gradient and hessian of the probabilities of variable at k being less than the rest.
        """
        q, d = dSigma_dx.shape[0], dmu_dx.shape[2]
        grad_a = np.zeros((q, d))
        tLk = Lk.T
        Dpk, Sigk_dx, mk_dx = np.zeros((q, d)), np.zeros((q, d, q, q)), np.zeros((q, d, q))

        # term A1: First term in Equation (6) in the paper
        grad_a[k, :] = - dmu_dx[k, k, :] * pk[k]  # q x q x d

        # compute gradient and hessian matrix of the CDF term pk.
        gradpk = Phi_gradient(bk - mk, np.zeros((q, 1)), Sigk)
        hesspk = Phi_hessian(bk, mk, Sigk, gradient=gradpk)
        for l, j in itertools.product(range(q), range(d)):
            Sigk_dx[l, j, :, :] = Lk @ dSigma_dx[:, :, l, j] @ tLk
            mk_dx[l, j, :] = Lk @ dmu_dx[:, l, j]
            Dpk[l, j] = 0.5 * np.sum(hesspk * Sigk_dx[l, j, :, :]) - gradpk[None, :] @ mk_dx[l, j, :, None]

        # term A2: Second term in Equation (6) in the paper
        return grad_a + (y_minimum - mu[k]) * Dpk, mk_dx, Sigk_dx, gradpk, hesspk

    def _gradient_of_the_acquisition_second_term(self, mu: np.ndarray, dmu_dx: np.ndarray, dSigma_dx: np.ndarray,
                                                 y_minimum: float, pk: np.ndarray, k: int, bk: np.ndarray,
                                                 mk: np.ndarray, Sigk: np.ndarray, mk_dx: np.ndarray,
                                                 Sigk_dx: np.ndarray, symmetric_term: np.ndarray,
                                                 gradpk: np.ndarray, hesspk: np.ndarray) -> np.ndarray:
        """
        Helper function for computing the second term of the gradient of the Acquisition (2-4 rows of Equation (6))

        :param mu: Prediction mean at locations where the evaluation is done.
        :param dmu_dx: Prediction mean gradient at locations where the evaluation is done.
        :param dSigma_dx: Prediction covariance gradient at locations where the evaluation is done.
        :param y_minimum: The best value evaluated so far by the black box function.
        :param pk: Probabilities of each random variable being smaller than the rest.
        :param k: Index of the assumed winner, as in Y_k < Y_i for all i not k
        :param bk: Vector b^{k} in the Equation (6) in the paper.
        :param mk: Mean of Y - Y_min given that Y_k is smaller than any other Y_i, i not k.
        :param Sigk: Covariance of Y - Y_min given that Y_k is smaller than any other Y_i, i not k.
        :param mk_dx: Gradient of mk.
        :param Sigk_dx: Gradient of Sigk.
        :param symmetric_term: A helper matrix generated by the acquistion function.
        :param gradpk: Gradient of vector pk.
        :param hesspk: Hessian of vecotr pk.
        :return: Gradient of the first term of the multipoint Expected Improvement.
        """
        q, d = dSigma_dx.shape[0], dmu_dx.shape[2]
        B = np.zeros((q, d))

        # term B (Rows 2-4 in Equation (6) in the paper)
        if self.fast_compute:
            # Numerical approximation through

            gradpk1 = Phi_gradient(bk - mk + Sigk[:, k, None] * self.eps, np.zeros((q, 1)), Sigk)
            hesspk1 = Phi_hessian(bk + Sigk[:, k, None] * self.eps, mk, Sigk, gradient=gradpk1)
            for l, j in itertools.product(range(q), range(d)):
                f1 = (-(mk_dx[None, l, j, :] @ gradpk1[:, None])
                      + self.eps * (Sigk_dx[None, l, j, :, k] @ gradpk1)
                      + 0.5 * np.sum(Sigk_dx[l, j, :, :] * hesspk1))
                f = -(mk_dx[None, l, j, :] @ gradpk[:, None]) + 0.5 * np.sum(Sigk_dx[l, j, :, :] * hesspk)
                B[l, j] = 1.0 / self.eps * (f1 - f)
        else:
            B1, B2, B3 = np.zeros((q, d)), np.zeros((q, d)), np.zeros((q, d))
            for i in range(q):

                # Assign helper variables needed by the gradients (See equation (6) for details)
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
                GPhi_ik = Phi_gradient(ck_pi, np.zeros((q - 1, 1)), Sigk_pi)
                HPhi_ik = Phi_hessian(ck_pi, np.zeros((q - 1, 1)), Sigk_pi, GPhi_ik)
                Sigk_mi = Sigk[ineq, i, None]

                # Compute the terms pf the gradient
                for l, j in itertools.product(range(q), range(d)):
                    # B1: Second row of Equation (6) in the paper
                    B1[l, j] = B1[l, j] + Sigk_dx_ik[l, j] * phi_ik * Phi_ik

                    # B2: Third row of Equation (6) in the paper
                    B2[l, j] = B2[l, j] + Sigk_ik * (mk_dx_i[l, j] * dphi_ik_dm
                                                     + dphi_ik_dSig * Sigk_dx_ii[l, j]) * Phi_ik

                    # B3: Fourth row of Equation (6) in the paper
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
            B = B1 + B2 + B3
        return B


def get_covariance_given_smallest(Sigma: np.ndarray, k: int) -> np.ndarray:
    """
    Compute covariance of Y - Y_min given that Y_k is smaller than any other Y_i, i not k

    :param Sigma: Covariance of the (not conditioned) multivariate gaussian.
    :param k: Index of random variable.
    :return: Covariance of Y - Y_min given that Y_k is smaller than any other Y_i
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


def get_covariance_given_value_of_i(Sigma: np.ndarray, i: int) -> np.ndarray:
    """
    Covariances of variables k knowing the value of variable i.

    :param Sigma: Covariance of the multivariate Gaussian.
    :param i: Known variable.
    :return: Covariance of Y given that Y_i is known
    """
    result = np.zeros_like(Sigma)
    neqi = [j for j in range(Sigma.shape[0]) if j is not i]  # Everything else but i
    for u, v in itertools.product(neqi, neqi):
        result[u, v] = Sigma[u, v] - Sigma[u, i] * Sigma[v, i] / Sigma[i, i]
    return result[neqi, :][:, neqi]


def get_correlations_given_value_of_i(b: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, i: int) -> np.ndarray:
    """
    Partial correlations of variables k knowing the value of variable i.

    :param b: Known values.
    :param mu: Mean of the multivariate Gaussian knowing i.
    :param Sigma: Covariance of the multivariate gaussian knowing i.
    :param i: known variable.
    :return: Partial correlations between all variables knowing i.
    """
    Sigmai = Sigma[i, :] / Sigma[i, i]
    result = (b - mu) - (b[i] - mu[i]) * Sigmai
    return result[np.arange(Sigma.shape[0]) != i]


def decompose_mvn(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, k: list) -> np.ndarray:
    """
    Decompose Multivariate normal to probability of some random variables being smaller than the rest and multiply this
    with the marginal probability distribution of those random variables.

    :param x: Location of decomposition.
    :param mu: Mean of the multivariate Gaussian.
    :param Sigma: Covariance of the multivariate gaussian.
    :param k: Indices of decomposition.
    :return: Weighted probabilities of each variable being smaller than the rest.
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


def Phi_gradient(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Compute gradient of CDF of multivariate Gaussian distribution.

    :param x: Location where the gradient is evaluated.
    :param mu: Mean of the multivariate Gaussian.
    :param Sigma: Covariance of the multivariate Gaussian.
    :return: Gradient of the CDF of multivariate Gaussian.
    """
    return np.array([decompose_mvn(x, mu, Sigma, [i]) for i in range(Sigma.shape[0])])


def Phi_hessian(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, gradient: np.ndarray=None) -> np.ndarray:
    """
    Compute hessian matrix of CDF of multivariate Gaussian distribution.

    :param x: Location where the Hessian is evaluated.
    :param mu: Mean of the multivariate Gaussian.
    :param Sigma: Covariance of the multivariate Gaussian.
    :param gradient: Gradient of the multivariate Gaussian at x.
    :return: Hessian of the CDF of multivariate Gaussian.
    """
    q = Sigma.shape[0]
    if (q == 1):
        res = - (x - mu) / Sigma * scipy.stats.norm.pdf(x, mu, np.sqrt(Sigma))
    else:
        res = np.zeros((q, q))
    for i in range(q - 1):
        for j in range(i + 1, q):
            res[i, j] = decompose_mvn(x, mu, Sigma, [i, j])

    res = res + res.T  # Hessian matrix is symmetric
    # diagonal terms can be computed with the gradient of CDF and the other hessian terms
    if gradient is None:
        res = res - np.diag(((x - mu).flatten() * Phi_gradient(x, mu, Sigma)[:, None]
                             + np.diag(Sigma @ res).flatten()) / np.diag(Sigma).flatten())
    else:
        res = res - np.diag(((x - mu).flatten() * gradient + np.diag(Sigma @ res).flatten()) / np.diag(Sigma).flatten())
    return res
