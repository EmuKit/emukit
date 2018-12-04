# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import mock

from emukit.quadrature.interfaces.base_gp import IBaseGaussianProcess
from emukit.quadrature.kernels.integral_bounds import IntegralBounds
from emukit.quadrature.methods.warped_bq_model import WarpedBayesianQuadratureModel
from emukit.quadrature.kernels.quadrature_kernels import QuadratureKernel
from emukit.core.continuous_parameter import ContinuousParameter


def test_warped_bq_model():
    X_train = np.random.rand(5, 2)
    Y_train = np.random.rand(5, 1)

    mock_cparams = mock.create_autospec(ContinuousParameter)
    mock_bounds = mock.create_autospec(IntegralBounds)
    mock_bounds.convert_to_list_of_continuous_parameters.return_value = 2 * [mock_cparams]
    mock_kern = mock.create_autospec(QuadratureKernel, integral_bounds=mock_bounds)
    mock_gp = mock.create_autospec(IBaseGaussianProcess, kern=mock_kern, X=X_train, Y=Y_train)
    method = WarpedBayesianQuadratureModel(base_gp=mock_gp)

    assert np.all(method.X == X_train)
    assert np.all(method.Y == Y_train)
    assert method.integral_bounds == mock_bounds
    assert method.integral_parameters == 2 * [mock_cparams]
