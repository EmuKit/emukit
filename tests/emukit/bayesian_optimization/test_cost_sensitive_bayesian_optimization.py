# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from emukit.bayesian_optimization.loops.cost_sensitive_bayesian_optimization_loop import (
    CostSensitiveBayesianOptimizationLoop,
)
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.model_wrappers import SimpleGaussianProcessModel


def test_cost_sensitive_bayesian_optimization_loop():
    space = ParameterSpace([ContinuousParameter("x", 0, 1)])

    x_init = np.random.rand(10, 1)

    def function_with_cost(x):
        return np.sin(x), x

    user_fcn = UserFunctionWrapper(function_with_cost, extra_output_names=["cost"])

    y_init, cost_init = function_with_cost(x_init)

    model_objective = SimpleGaussianProcessModel(x_init, y_init)
    model_cost = SimpleGaussianProcessModel(x_init, cost_init)

    loop = CostSensitiveBayesianOptimizationLoop(space, model_objective, model_cost)
    loop.run_loop(user_fcn, 10)

    assert loop.loop_state.X.shape[0] == 20
    assert loop.loop_state.cost.shape[0] == 20
