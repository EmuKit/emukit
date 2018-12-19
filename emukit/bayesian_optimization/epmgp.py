# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy import special


# some variables
sq2 = np.sqrt(2)
eps = np.finfo(np.float32).eps
l2p = np.log(2) + np.log(np.pi)


def joint_min(mu: np.ndarray, var: np.ndarray, with_derivatives: bool=False) -> np.ndarray:
    """
    Computes the probability of every given point to be the minimum
    based on the EPMGP[1] algorithm.
    [1] J. Cunningham, P. Hennig, and S. Lacoste-Julien.
    Gaussian probabilities and expectation propagation.
    under review. Preprint at arXiv, November 2011.

    :param mu: Mean value of each of the N points, dims (N,).
    :param var: Covariance matrix for all points, dims (N, N).
    :param with_derivatives: If True than also the gradients are computed.
    :returns: pmin distribution, dims (N,1).
    """

    logP = np.zeros(mu.shape)
    D = mu.shape[0]
    if with_derivatives:
        dlogPdMu = np.zeros((D, D))
        dlogPdSigma = np.zeros((D, int(0.5 * D * (D + 1))))
        dlogPdMudMu = np.zeros((D, D, D))
    for i in range(mu.shape[0]):

        # logP[k] ) self._min_factor(mu, var, 0)
        a = min_factor(mu, var, i)

        logP[i] = next(a)
        if with_derivatives:
            dlogPdMu[i, :] = next(a).T
            dlogPdMudMu[i, :, :] = next(a)
            dlogPdSigma[i, :] = next(a).T

    logP[np.isinf(logP)] = -500
    # re-normalize at the end, to smooth out numerical imbalances:
    logPold = logP
    Z = np.sum(np.exp(logPold))
    maxLogP = np.max(logP)
    s = maxLogP + np.log(np.sum(np.exp(logP - maxLogP)))
    s = maxLogP if np.isinf(s) else s

    logP = logP - s
    if not with_derivatives:
        return logP

    dlogPdMuold = dlogPdMu
    dlogPdSigmaold = dlogPdSigma
    dlogPdMudMuold = dlogPdMudMu
    # adjust derivatives, too. This is a bit tedious.
    Zm = sum(np.rot90((np.exp(logPold) * np.rot90(dlogPdMuold, 1)), 3)) / Z
    Zs = sum(np.rot90((np.exp(logPold) * np.rot90(dlogPdSigmaold, 1)), 3)) / Z

    dlogPdMu = dlogPdMuold - Zm
    dlogPdSigma = dlogPdSigmaold - Zs

    ff = np.einsum('ki,kj->kij', dlogPdMuold, dlogPdMuold)
    gg = np.einsum('kij,k->ij', dlogPdMudMuold + ff, np.exp(logPold)) / Z
    Zij = Zm.T * Zm
    adds = np.reshape(-gg + Zij, (1, D, D))
    dlogPdMudMu = dlogPdMudMuold + adds
    return logP, dlogPdMu, dlogPdSigma, dlogPdMudMu


def min_factor(Mu, Sigma, k, gamma=1):
    D = Mu.shape[0]
    logS = np.zeros((D - 1,))
    # mean time first moment
    MP = np.zeros((D - 1,))

    # precision, second moment
    P = np.zeros((D - 1,))

    M = np.copy(Mu)
    V = np.copy(Sigma)
    b = False
    d = np.NaN
    for count in range(50):
        diff = 0
        for i in range(D - 1):
            l = i if i < k else i + 1  # noqa: E741 to be consistent with paper notation
            try:
                M, V, P[i], MP[i], logS[i], d = lt_factor(k, l, M, V,
                                                          MP[i], P[i], gamma)
            except Exception as e:
                raise

            if np.isnan(d):
                break
            diff += np.abs(d)
        if np.isnan(d):
            break
        if np.abs(diff) < 0.001:
            b = True
            break
    if np.isnan(d):
        logZ = -np.Infinity
        yield logZ
        dlogZdMu = np.zeros((D, 1))
        yield dlogZdMu

        dlogZdMudMu = np.zeros((D, D))
        yield dlogZdMudMu
        dlogZdSigma = np.zeros((int(0.5 * (D * (D + 1))), 1))
        yield dlogZdSigma
        mvmin = [Mu[k], Sigma[k, k]]
        yield mvmin
    else:
        # evaluate log Z:
        C = np.eye(D) / sq2
        C[k, :] = -1 / sq2
        C = np.delete(C, k, 1)

        R = np.sqrt(P.T) * C
        r = np.sum(MP.T * C, 1)
        mp_not_zero = np.where(MP != 0)
        mpm = MP[mp_not_zero] * MP[mp_not_zero] / P[mp_not_zero]
        mpm = sum(mpm)

        s = sum(logS)
        IRSR = (np.eye(D - 1) + np.dot(np.dot(R.T, Sigma), R))
        rSr = np.dot(np.dot(r.T, Sigma), r)
        A = np.dot(R, np.linalg.solve(IRSR, R.T))

        A = 0.5 * (A.T + A)  # ensure symmetry.
        b = (Mu + np.dot(Sigma, r))
        Ab = np.dot(A, b)
        try:
            cIRSR = np.linalg.cholesky(IRSR)
        except np.linalg.LinAlgError:
            try:
                cIRSR = np.linalg.cholesky(IRSR + 1e-10 * np.eye(IRSR.shape[0]))
            except np.linalg.LinAlgError:
                cIRSR = np.linalg.cholesky(IRSR + 1e-6 * np.eye(IRSR.shape[0]))
        dts = 2 * np.sum(np.log(np.diagonal(cIRSR)))
        logZ = 0.5 * (rSr - np.dot(b.T, Ab) - dts) + np.dot(Mu.T, r) + s - 0.5 * mpm
        yield logZ
        btA = np.dot(b.T, A)

        dlogZdMu = r - Ab
        yield dlogZdMu
        dlogZdMudMu = -A
        yield dlogZdMudMu
        dlogZdSigma = -A - 2 * np.outer(r, Ab.T) + np.outer(r, r.T) \
                      + np.outer(btA.T, Ab.T)
        dlogZdSigma2 = np.zeros_like(dlogZdSigma)
        np.fill_diagonal(dlogZdSigma2, np.diagonal(dlogZdSigma))
        dlogZdSigma = 0.5 * (dlogZdSigma + dlogZdSigma.T - dlogZdSigma2)
        dlogZdSigma = np.rot90(dlogZdSigma, k=2)[np.triu_indices(D)][::-1]
        yield dlogZdSigma


def lt_factor(s, l, M, V, mp, p, gamma):
    cVc = (V[l, l] - 2 * V[s, l] + V[s, s]) / 2.0
    Vc = (V[:, l] - V[:, s]) / sq2
    cM = (M[l] - M[s]) / sq2
    cVnic = np.max([cVc / (1 - p * cVc), 0])
    cmni = cM + cVnic * (p * cM - mp)
    z = cmni / np.sqrt(cVnic + 1e-25)
    if np.isnan(z):
        z = -np.inf
    e, lP, exit_flag = log_relative_gauss(z)
    if exit_flag == 0:
        alpha = e / np.sqrt(cVnic)
        # beta  = alpha * (alpha + cmni / cVnic);
        # r     = beta * cVnic / (1 - cVnic * beta);
        beta = alpha * (alpha * cVnic + cmni)
        r = beta / (1 - beta)
        # new message
        pnew = r / cVnic
        mpnew = r * (alpha + cmni / cVnic) + alpha

        # update terms
        dp = np.max([-p + eps, gamma * (pnew - p)])  # at worst, remove message
        dmp = np.max([-mp + eps, gamma * (mpnew - mp)])
        d = np.max([dmp, dp])  # for convergence measures

        pnew = p + dp
        mpnew = mp + dmp
        # project out to marginal
        Vnew = V - dp / (1 + dp * cVc) * np.outer(Vc, Vc)

        Mnew = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc
        if np.any(np.isnan(Vnew)):
            raise Exception("an error occurs while running expectation "
                            "propagation in entropy search. "
                            "Resulting variance contains NaN")
        # % there is a problem here, when z is very large
        logS = lP - 0.5 * (np.log(beta) - np.log(pnew) - np.log(cVnic)) \
               + (alpha * alpha) / (2 * beta) * cVnic

    elif exit_flag == -1:
        d = np.NAN
        Mnew = 0
        Vnew = 0
        pnew = 0
        mpnew = 0
        logS = -np.Infinity
    elif exit_flag == 1:
        d = 0
        # remove message from marginal:
        # new message
        pnew = 0
        mpnew = 0
        # update terms
        dp = -p  # at worst, remove message
        dmp = -mp
        d = max([dmp, dp])  # for convergence measures
        # project out to marginal
        Vnew = V - dp / (1 + dp * cVc) * (np.outer(Vc, Vc))
        Mnew = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc
        logS = 0
    return Mnew, Vnew, pnew, mpnew, logS, d


def log_relative_gauss(z):
    """
    log_relative_gauss
    """
    if z < -6:
        return 1, -1.0e12, -1
    if z > 6:
        return 0, 0, 1
    else:
        logphi = -0.5 * (z * z + l2p)
        logPhi = np.log(.5 * special.erfc(-z / sq2))
        e = np.exp(logphi - logPhi)
    return e, logPhi, 0
