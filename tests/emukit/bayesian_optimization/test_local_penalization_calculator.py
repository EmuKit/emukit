# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import GPy
import mock
import numpy as np
import pytest

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.local_penalization_calculator import LocalPenalizationPointCalculator
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.interfaces import IModel
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.model_wrappers import GPyModelWrapper


def test_local_penalization():
    parameter_space = ParameterSpace([ContinuousParameter("x", 0, 1)])
    acquisition_optimizer = GradientAcquisitionOptimizer(parameter_space)
    x_init = np.random.rand(5, 1)
    y_init = np.random.rand(5, 1)

    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = GPyModelWrapper(gpy_model)
    acquisition = ExpectedImprovement(model)
    batch_size = 5
    lp_calc = LocalPenalizationPointCalculator(acquisition, acquisition_optimizer, model, parameter_space, batch_size)

    loop_state = create_loop_state(x_init, y_init)
    new_points = lp_calc.compute_next_points(loop_state)
    assert new_points.shape == (batch_size, 1)


def test_local_penalization_requires_gradients():
    parameter_space = ParameterSpace([ContinuousParameter("x", 0, 1)])
    acquisition_optimizer = GradientAcquisitionOptimizer(parameter_space)

    model = mock.create_autospec(IModel)

    acquisition = ExpectedImprovement(model)
    batch_size = 5

    with pytest.raises(ValueError):
        LocalPenalizationPointCalculator(acquisition, acquisition_optimizer, model, parameter_space, batch_size)
