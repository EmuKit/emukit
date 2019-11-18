# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ..interfaces.base_gp import IBaseGaussianProcess
from .bounded_sqrt_model import BoundedBQSqrtTransformLinearApproxBQModel


class WSABIL(BoundedBQSqrtTransformLinearApproxBQModel):
    """
     WSABI-L (Warped Sequential Active Bayesian Integration with linear approximation)

    Gunter et al. 2014
    Sampling for Inference in Probabilistic Models with Fast Bayesian Quadrature
    Advances in Neural Information Processing Systems (NeurIPS), 27, pp. 2789–2797.

    WSABI must be used with the RBF kernel and the Gaussian integration measure. This means that the kernel of base_gp
    must be of type QuadratureRBFIsoGaussMeasure.

    The linear approximation is described in Gunter et al. in section 3.1, equations 9 and 10.
    """

    def __init__(self, base_gp: IBaseGaussianProcess, X: np.ndarray, Y: np.ndarray, adapt_offset: bool=True):
        """
        :param base_gp: a model derived from BaseGaussianProcess. Must use QuadratureRBFIsoGaussMeasure as kernel.
        :param X: the initial locations of integrand evaluations
        :param Y: the values of the integrand at Y
        :param adapt_offset: If True, offset of transformation will be adapted according to 0.8 x min(Y) as in
        Gunter et al.. If False the offset will be fixed to zero. Default is True.
        the offset will bet set to zero.
        """
        self.adapt_offset = adapt_offset
        if adapt_offset:
            bound = self._compute_offset(X, Y)
        else:
            bound = 0.
        super(WSABIL, self).__init__(base_gp=base_gp, X=X, Y=Y, bound=bound, lower_bounded=True)

    def _compute_offset(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        the value for the offset is given in Gunter et al. 2014 on page 3 in the footnote

        :param X: observation locations, shape (num_points, dim)
        :param Y: values of observations, shape (num_points, 1)
        :return: the scalar offset
        """
        offset = 0.8 * min(Y)[0]
        return offset

    def update_parameters(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        computes offset and sets new offset.
        :param X: observation locations, shape (num_points, dim)
        :param Y: values of observations, shape (num_points, 1)
        """
        if self.adapt_offset:
            self.bound = self._compute_offset(X, Y)
