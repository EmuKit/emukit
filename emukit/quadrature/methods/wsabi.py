# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Optional

from ..interfaces.base_gp import IBaseGaussianProcess
from .bounded_bq_model import BoundedBQModel


class WSABIL(BoundedBQModel):
    """WSABI-L (Warped Sequential Active Bayesian Integration with linear approximation).

    Gunter et al. 2014
    Sampling for Inference in Probabilistic Models with Fast Bayesian Quadrature
    Advances in Neural Information Processing Systems (NeurIPS), 27, pp. 2789–2797.

    WSABI must be used with the RBF kernel and the Gaussian integration measure. This means that the kernel of base_gp
    must be of type QuadratureRBFIsoGaussMeasure.

    The linear approximation is described in Gunter et al. in section 3.1, equations 9 and 10.
    """

    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray, adapt_alpha: bool=True):
        """
        :param base_gp: a model derived from BaseGaussianProcess. Must use QuadratureRBFIsoGaussMeasure as kernel.
        :param X: the initial locations of integrand evaluations.
        :param Y: the values of the integrand at Y.
        :param adapt_alpha: If ``True``, the offset :math:`\alpha` will be adapted according to :math:`0.8 min(Y)` as
               in Gunter et al., page 3, footnote. If ``False`` :math:`\alpha` will be fixed to a small value for
               numerical stability. Default is ``True``.
        """
        self._small_alpha = 1e-8  # only used if alpha is not adapted
        alpha = self._compute_alpha(X, Y)
        super(WSABIL, self).__init__(base_gp=base_gp, X=X, Y=Y, bound=alpha, lower_bounded=True)
        self.adapt_offset = adapt_alpha

    def _compute_alpha(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        The value for the offset is given in Gunter et al. 2014 on page 3 in the footnote.
        Will be computed from the data Y, only of ``self.adapt_offset`` is ``True``. Otherwise the offset is small
        number for numerical stability.

        :param X: observation locations, shape (num_points, input_dim)
        :param Y: values of observations, shape (num_points, 1)
        :return: the scalar offset.
        """
        if self.adapt_offset:
            return 0.8 * min(Y)[0]
        return self._small_alpha

    def update_parameters(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Computes and sets the offset :math:`\alpha`.
        :param X: observation locations, shape (num_points, dim)
        :param Y: values of observations, shape (num_points, 1)
        """
        self._warping.update_parameters(bound=self._compute_alpha(X, Y))
