# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from emukit.examples.fabolas import FabolasModel


@pytest.fixture
def model():
    rng = np.random.RandomState(42)
    x_init = rng.rand(5, 2)
    s_min = 10
    s_max = 10000
    s = np.random.uniform(s_min, s_max, x_init.shape[0])
    x_init = np.concatenate((x_init, s[:, None]), axis=1)
    y_init = rng.rand(5, 1)

    model = FabolasModel(X_init=x_init, Y_init=y_init, s_min=s_min, s_max=s_max)
    return model


def test_predict_shape(model):
    rng = np.random.RandomState(43)

    x_test = rng.rand(10, 2)
    s = np.random.uniform(model.s_min, model.s_max, x_test.shape[0])
    x_test = np.concatenate((x_test, s[:, None]), axis=1)
    m, v = model.predict(x_test)

    assert m.shape == (10, 1)
    assert v.shape == (10, 1)


def test_update_data(model):
    rng = np.random.RandomState(43)
    x_new = rng.rand(5, 2)
    s = np.random.uniform(model.s_min, model.s_max, x_new.shape[0])
    x_new = np.concatenate((x_new, s[:, None]), axis=1)
    y_new = rng.rand(5, 1)
    model.set_data(x_new, y_new)

    assert model.X.shape == x_new.shape
    assert model.Y.shape == y_new.shape
