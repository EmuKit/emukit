# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import GPy
import numpy as np
import pytest

import emukit.test_functions
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.benchmarking.loop_benchmarking.benchmarker import Benchmarker
from emukit.benchmarking.loop_benchmarking.metrics import MeanSquaredErrorMetric, MinimumObservedValueMetric, TimeMetric
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.model_wrappers import GPyModelWrapper


@pytest.fixture
def loops():
    space = ParameterSpace([ContinuousParameter("x", 0, 1)])

    def make_loop(loop_state):
        gpy_model = GPy.models.GPRegression(loop_state.X, loop_state.Y)
        model = GPyModelWrapper(gpy_model)
        return BayesianOptimizationLoop(space, model)

    return [("GP", make_loop)]


def test_benchmarker_runs(loops):
    test_fcn, parameter_space = emukit.test_functions.forrester_function()

    x_test = np.random.rand(50, 1)
    benchmark = Benchmarker(
        loops,
        test_fcn,
        parameter_space,
        [MinimumObservedValueMetric(), TimeMetric(), MeanSquaredErrorMetric(x_test, test_fcn(x_test))],
    )
    results = benchmark.run_benchmark()


def test_non_unique_metric_names_fail(loops):
    test_fcn, parameter_space = emukit.test_functions.forrester_function()
    with pytest.raises(ValueError):
        Benchmarker(
            loops, test_fcn, parameter_space, [MinimumObservedValueMetric("x"), MinimumObservedValueMetric("x")]
        )
