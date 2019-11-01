# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import GPy
from typing import List, Tuple, Optional

from emukit.quadrature.methods.vanilla_bq import VanillaBayesianQuadrature
from emukit.quadrature.loop import VanillaBayesianQuadratureLoop
from emukit.quadrature.kernels.integration_measures import IntegrationMeasure
from emukit.model_wrappers.gpy_quadrature_wrappers import create_emukit_model_from_gpy_model


def create_vanilla_bq_loop_with_rbf_kernel(X: np.ndarray, Y: np.ndarray,
                                           integral_bounds: Optional[List[Tuple[float, float]]],
                                           measure: Optional[IntegrationMeasure],
                                           rbf_lengthscale: float=1.0, rbf_variance: float=1.0) \
        -> VanillaBayesianQuadratureLoop:
    """

    :param X: initial training point locations, shape (n_points, input_dim)
    :param Y:  initial training point function values, shape (n_points, 1)
    :param integral_bounds: List of input_dim tuples, where input_dim is the dimensionality of the integral
    and the tuples contain the lower and upper bounds of the integral i.e.,
    [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]. None means infinite bounds.
    :param measure: the integration measure. None means the standard Lebesgue measure is used.
    :param rbf_lengthscale: the lengthscale of the rbf kernel, defaults to 1.
    :param rbf_variance: the variance of the rbf kernel, defaults to 1.
    :return: The vanilla BQ loop
    """

    if (integral_bounds is not None) and (len(integral_bounds) != X.shape[1]):
            D_bounds = len(integral_bounds)
            input_dim = X.shape[1]
            raise ValueError(
                "dimension of integral bounds provided ({}) does not match the input dimension of X ({}).".format(
                    D_bounds, input_dim))
    if rbf_lengthscale <= 0:
        raise ValueError("rbf lengthscale must be positive. The current value is {}.".format(rbf_lengthscale))
    if rbf_variance <= 0:
        raise ValueError("rbf variance must be positive. The current value is {}.".format(rbf_variance))

    gpy_model = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=X.shape[1],
                                                                      lengthscale=rbf_lengthscale,
                                                                      variance=rbf_variance))

    emukit_model = create_emukit_model_from_gpy_model(gpy_model=gpy_model,
                                                      integral_bounds=integral_bounds,
                                                      measure=measure)
    emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=X, Y=Y)
    emukit_loop = VanillaBayesianQuadratureLoop(model=emukit_method)
    return emukit_loop
