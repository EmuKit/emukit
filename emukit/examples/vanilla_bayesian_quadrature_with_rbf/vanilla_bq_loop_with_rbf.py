# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import GPy
import numpy as np

from emukit.model_wrappers.gpy_quadrature_wrappers import create_emukit_model_from_gpy_model
from emukit.quadrature.loop import VanillaBayesianQuadratureLoop
from emukit.quadrature.measures import IntegrationMeasure
from emukit.quadrature.methods.vanilla_bq import VanillaBayesianQuadrature
from emukit.quadrature.typing import BoundsType


def create_vanilla_bq_loop_with_rbf_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    integral_bounds: Optional[BoundsType] = None,
    measure: Optional[IntegrationMeasure] = None,
    rbf_lengthscale: float = 1.0,
    rbf_variance: float = 1.0,
) -> VanillaBayesianQuadratureLoop:
    """Creates a quadrature loop with a standard (vanilla) model.

    :param X: Initial training point locations, shape (n_points, input_dim).
    :param Y: Initial training point function values, shape (n_points, 1).
    :param integral_bounds: List of d tuples, where d is the dimensionality of the integral and the tuples contain the
                            lower and upper bounds of the integral
                            i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_d, ub_d)].
                            Only used if ``measure`` is not given in which case the unnormalized Lebesgue measure is used.
    :param measure: An integration measure. Either ``measure`` or ``integral_bounds`` must be given.
                    If both ``integral_bounds`` and ``measure`` are given, ``integral_bounds`` is disregarded.
    :param rbf_lengthscale: The lengthscale of the rbf kernel, defaults to 1.
    :param rbf_variance: The variance of the rbf kernel, defaults to 1.
    :return: The vanilla BQ loop.

    """

    if measure is not None and measure.input_dim != X.shape[1]:
        raise ValueError(
            f"Dimensionality of measure ({measure.input_dim}) does not match the dimensionality of "
            f"the data ({X.shape[1]})."
        )

    if (integral_bounds is not None) and (len(integral_bounds) != X.shape[1]):
        raise ValueError(
            f"Dimension of integral bounds ({len(integral_bounds)}) does not match the input dimension "
            f"of X ({X.shape[1]})."
        )

    if rbf_lengthscale <= 0:
        raise ValueError(f"rbf lengthscale must be positive. The current value is {rbf_lengthscale}.")
    if rbf_variance <= 0:
        raise ValueError(f"rbf variance must be positive. The current value is {rbf_variance}.")

    gpy_model = GPy.models.GPRegression(
        X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=X.shape[1], lengthscale=rbf_lengthscale, variance=rbf_variance)
    )

    # This function handles the omittion if the integral bounds in case measure is also given.
    emukit_model = create_emukit_model_from_gpy_model(
        gpy_model=gpy_model, integral_bounds=integral_bounds, measure=measure
    )
    emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=X, Y=Y)
    emukit_loop = VanillaBayesianQuadratureLoop(model=emukit_method)
    return emukit_loop
