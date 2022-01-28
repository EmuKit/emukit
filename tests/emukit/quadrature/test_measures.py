# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from math import isclose

import numpy as np
import pytest

from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure, UniformMeasure

REL_TOL = 1e-5
ABS_TOL = 1e-4


def test_uniform_measure_shapes():

    N = 5
    bounds = [(-1, 1), (0, 2), (1.3, 5.0)]
    D = len(bounds)
    x = np.reshape(np.random.randn(D * N), [N, D])

    measure = UniformMeasure(bounds)

    bounds = measure.get_box()
    assert len(bounds) == D
    assert len(bounds[0]) == 2

    res = measure.compute_density(x)
    assert res.shape == (N,)

    # sampling capabilities
    assert measure.can_sample
    res = measure.get_samples(N)
    assert res.shape == (N, D)


def test_uniform_measure_wrong_bounds():
    bounds = [(-1, 1), (3, 2), (1.3, 5.0)]

    with pytest.raises(ValueError):
        UniformMeasure(bounds)


def test_uniform_measure_gradients():
    measure_bounds = [(-1, 2), (0, 1)]
    measure = UniformMeasure(bounds=measure_bounds)

    N = 3
    D = len(measure_bounds)
    x = np.reshape(np.random.randn(D * N), [N, D])
    _check_grad(measure, x)


def test_iso_gauss_measure_shapes():
    D = 4
    N = 5
    x = np.reshape(np.random.randn(D * N), [N, D])
    measure = IsotropicGaussianMeasure(mean=np.ones(D), variance=1.0)

    bounds = measure.get_box()
    assert len(bounds) == D
    assert len(bounds[0]) == 2
    assert measure.num_dimensions == D

    res = measure.compute_density(x)
    assert res.shape == (N,)

    # sampling capabilities
    assert measure.can_sample
    res = measure.get_samples(N)
    assert res.shape == (N, D)


def test_iso_gauss_measure_invalid_input():
    wrong_variance = -2.0
    mean_wrong_dim = np.ones([3, 1])
    mean_wrong_type = 0.0

    with pytest.raises(ValueError):
        IsotropicGaussianMeasure(mean=np.ones(3), variance=wrong_variance)

    with pytest.raises(TypeError):
        IsotropicGaussianMeasure(mean=mean_wrong_type, variance=1.0)

    with pytest.raises(ValueError):
        IsotropicGaussianMeasure(mean=mean_wrong_dim, variance=1.0)


def test_iso_gauss_measure_gradients():
    D = 2
    measure = IsotropicGaussianMeasure(mean=np.random.randn(D), variance=np.random.randn() ** 2)

    N = 3
    x = np.reshape(np.random.randn(D * N), [N, D])
    _check_grad(measure, x)


def _compute_numerical_gradient(m, x, eps=1e-6):
    f = m.compute_density(x)
    grad = m.compute_density_gradient(x)

    grad_num = np.zeros(grad.shape)
    for d in range(x.shape[1]):
        x_tmp = x.copy()
        x_tmp[:, d] = x_tmp[:, d] + eps
        f_tmp = m.compute_density(x_tmp)
        grad_num_d = (f_tmp - f) / eps
        grad_num[:, d] = grad_num_d
    return grad, grad_num


def _check_grad(aq, x):
    grad, grad_num = _compute_numerical_gradient(aq, x)
    isclose_all = 1 - np.array(
        [
            isclose(grad[i, j], grad_num[i, j], rel_tol=REL_TOL, abs_tol=ABS_TOL)
            for i in range(grad.shape[0])
            for j in range(grad.shape[1])
        ]
    )
    assert isclose_all.sum() == 0
