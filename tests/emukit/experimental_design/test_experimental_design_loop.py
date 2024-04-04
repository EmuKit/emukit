# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import GPy
import numpy as np
from numpy.testing import assert_array_equal

from emukit.core.continuous_parameter import ContinuousParameter
from emukit.core.loop import FixedIterationsStoppingCondition, UserFunctionWrapper
from emukit.core.parameter_space import ParameterSpace
from emukit.experimental_design import ExperimentalDesignLoop
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.model_wrappers import GPyModelWrapper


def f(x):
    return x**2


def test_loop_initial_state():
    x_init = np.random.rand(5, 1)
    y_init = np.random.rand(5, 1)

    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = GPyModelWrapper(gpy_model)
    space = ParameterSpace([ContinuousParameter("x", 0, 1)])

    exp_design = ExperimentalDesignLoop(space, model)

    # test loop state initialization
    assert_array_equal(exp_design.loop_state.X, x_init)
    assert_array_equal(exp_design.loop_state.Y, y_init)


def test_loop():
    n_iterations = 5

    x_init = np.random.rand(5, 1)
    y_init = np.random.rand(5, 1)

    # Make GPy model
    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = GPyModelWrapper(gpy_model)

    space = ParameterSpace([ContinuousParameter("x", 0, 1)])
    acquisition = ModelVariance(model)

    # Make loop and collect points
    exp_design = ExperimentalDesignLoop(space, model, acquisition)
    exp_design.run_loop(UserFunctionWrapper(f), FixedIterationsStoppingCondition(n_iterations))

    # Check we got the correct number of points
    assert exp_design.loop_state.X.shape[0] == 10
