# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import GPy
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, QuadratureRBFLebesgueMeasure, RBFGPy
from emukit.quadrature.loop import VanillaBayesianQuadratureLoop
from emukit.quadrature.measures import LebesgueMeasure
from emukit.quadrature.methods import VanillaBayesianQuadrature


def func(x):
    return np.ones((x.shape[0], 1))


@pytest.fixture
def loop():
    init_size = 5
    x_init = np.random.rand(init_size, 2)
    y_init = np.random.rand(init_size, 1)
    bounds = [(-1, 1), (0, 1)]

    gpy_model = GPy.models.GPRegression(
        X=x_init, Y=y_init, kernel=GPy.kern.RBF(input_dim=x_init.shape[1], lengthscale=1.0, variance=1.0)
    )
    emukit_measure = LebesgueMeasure.from_bounds(bounds, normalized=False)
    emukit_qrbf = QuadratureRBFLebesgueMeasure(RBFGPy(gpy_model.kern), measure=emukit_measure)
    emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)
    emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=x_init, Y=y_init)
    emukit_loop = VanillaBayesianQuadratureLoop(model=emukit_method)
    return emukit_loop, init_size, x_init, y_init


def test_vanilla_bq_loop(loop):
    emukit_loop, init_size, _, _ = loop
    num_iter = 5

    emukit_loop.run_loop(user_function=UserFunctionWrapper(func), stopping_condition=num_iter)

    assert emukit_loop.loop_state.X.shape[0] == num_iter + init_size
    assert emukit_loop.loop_state.Y.shape[0] == num_iter + init_size


def test_vanilla_bq_loop_initial_state(loop):
    emukit_loop, _, x_init, y_init = loop

    assert_array_equal(emukit_loop.loop_state.X, x_init)
    assert_array_equal(emukit_loop.loop_state.Y, y_init)
    assert emukit_loop.loop_state.iteration == 0
