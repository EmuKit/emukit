# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from scipy.optimize import check_grad

from emukit.bayesian_optimization.acquisitions.local_penalization import LocalPenalization
from emukit.core.interfaces import IModel

TOL = 1e-6


def test_penalization_function_shape():
    model = MockModel()
    lp = LocalPenalization(model)
    lp.update_batches(np.zeros((5, 1)), 1, -0.1)

    value = lp.evaluate(np.random.rand(10, 1))
    assert value.shape == (10, 1)


def test_penalization_function_gradients_shape():
    model = MockModel()
    lp = LocalPenalization(model)
    lp.update_batches(np.zeros((5, 2)), 1, -0.1)

    val, grad = lp.evaluate_with_gradients(np.random.rand(10, 2))
    assert grad.shape == (10, 2)
    assert val.shape == (10, 1)


def test_local_penalization_gradients_with_single_point_in_batch():
    np.random.seed(123)
    model = MockModel()
    lp = LocalPenalization(model)
    lp.update_batches(np.zeros((1, 1)), 1, -0.1)

    x0 = np.array([0.5])
    _check_grad(lp, TOL, x0)


def test_local_penalization_gradients_with_no_points_in_batch():
    np.random.seed(123)
    model = MockModel()
    lp = LocalPenalization(model)
    lp.update_batches(np.zeros((1, 1)), 1, -0.1)

    x0 = np.array([0.5])
    _check_grad(lp, TOL, x0)


def test_local_penalization_gradients_with_multiple_points_in_batch():
    np.random.seed(123)
    model = MockModel()
    lp = LocalPenalization(model)
    lp.update_batches(np.random.rand(5, 1), 1, -0.1)

    x0 = np.array([0.5])
    _check_grad(lp, TOL, x0)


def test_local_penalization_at_batch_point():
    # Test edge case where evaluating local penalization at a point already in the batch.
    # This can lead to divide by zero errors if not done correctly.

    np.random.seed(123)
    model = MockModel()
    lp = LocalPenalization(model)
    x_batch = np.random.rand(5, 1)
    lp.update_batches(x_batch, 1, -0.1)

    val, grad = lp.evaluate_with_gradients(x_batch)
    assert not np.any(np.isnan(grad))


class MockModel(IModel):
    def predict(self, X):
        return np.random.rand(X.shape[0], 1), np.random.rand(X.shape[0], 1)


def _check_grad(lp, tol, x0):
    grad_error = check_grad(
        lambda x: lp.evaluate_with_gradients(x[None, :])[0][0], lambda x: lp.evaluate_with_gradients(x[None, :])[1], x0
    )
    assert np.all(grad_error < tol)
