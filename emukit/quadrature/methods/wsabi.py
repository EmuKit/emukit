# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""WSABI model as in Gunter et al. 2014"""


import numpy as np

from ..interfaces.base_gp import IBaseGaussianProcess
from . import BoundedBayesianQuadrature


class WSABIL(BoundedBayesianQuadrature):
    r"""Warped Sequential Active Bayesian Integration with linear approximation (WSABI-L).

    The linear approximation is described in `[1]`_ in section 3.1, Eq (9) and (10).

    The offset :math:`\alpha` (notation from paper) will either be set to a small value if
    ``adapt_alpha`` is ``False``. Else it will be adapted according to
    :math:`0.8 \operatorname{min}(Y)` as in Gunter et al. 2014, page 3, footnote,
    where :math:`Y` are the collected integrand evaluations so far.

    .. _[1]:

    .. rubric:: References

    [1] Gunter et al. 2014 *Sampling for Inference in Probabilistic Models with Fast Bayesian Quadrature*,
    Advances in Neural Information Processing Systems (NeurIPS), 27, pp. 2789–2797.

    .. note::
        WSABI-L must be used with the RBF kernel and the Gaussian integration measure
        (See description of :attr:`base_gp` parameter). The loop must use the uncertainty
        sampling acquisition strategy.

    .. seealso::
        * :class:`emukit.quadrature.methods.BoundedBayesianQuadrature`
        * :class:`emukit.quadrature.acquisitions.UncertaintySampling`
        * :class:`emukit.quadrature.loop.WSABILLoop`

    :param base_gp: A standard Gaussian process.
                    Must use :class:`emukit.quadrature.kernels.QuadratureRBFGaussianMeasure` as kernel.
    :param X: The initial locations of integrand evaluations, shape (n_points, input_dim).
    :param Y: The values of the integrand at X, shape (n_points, 1).
    :param adapt_alpha: If ``True``, the offset :math:`\alpha` will be adapted. If ``False`` :math:`\alpha` will be
                        fixed to a small value for numerical stability. Default is ``True``.

    """

    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray, adapt_alpha: bool = True):
        self.adapt_alpha = adapt_alpha
        self._small_alpha = 1e-8  # only used if alpha is not adapted
        alpha = self._compute_alpha(X, Y)

        super(WSABIL, self).__init__(base_gp=base_gp, X=X, Y=Y, lower_bound=alpha)

    def _compute_alpha(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute the offset :math:`\alpha`.

        :param X: Observation locations, shape (n_points, input_dim)
        :param Y: Values of observations, shape (n_points, 1)
        :return: The offset :math:`\alpha`.
        """
        if self.adapt_alpha:
            return 0.8 * min(Y)[0]
        return self._small_alpha

    def compute_warping_params(self, X: np.ndarray, Y: np.ndarray) -> dict:
        r"""Computes the new :math:`\alpha` parameter from data.

        :param X: Observation locations, shape (n_points, input_dim).
        :param Y: Integrand observations at X, shape (n_points, 1).
        :return: Dictionary containing new value of :math:`\alpha`.
        """
        new_offset = self._compute_alpha(X, Y)
        return {"offset": new_offset}
