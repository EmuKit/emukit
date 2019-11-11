# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from emukit.quadrature.kernels.integration_measures import UniformMeasure, IsotropicGaussianMeasure


def test_uniform_measure_shapes():

    N = 5
    bounds = [(-1, 1), (0, 2), (1.3, 5.)]
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
    bounds = [(-1, 1), (3, 2), (1.3, 5.)]

    with pytest.raises(ValueError):
        UniformMeasure(bounds)


def test_iso_gauss_measure_shapes():
    D = 4
    N = 5
    x = np.reshape(np.random.randn(D * N), [N, D])
    measure = IsotropicGaussianMeasure(mean=np.ones(D), variance=1.)

    bounds = measure.get_box()
    assert len(bounds) == D
    assert len(bounds[0]) == 2
    assert measure.dim == D

    res = measure.compute_density(x)
    assert res.shape == (N, )

    # sampling capabilities
    assert measure.can_sample
    res = measure.get_samples(N)
    assert res.shape == (N, D)


def test_iso_gauss_measure_invalid_input():
    wrong_variance = -2.
    mean_wrong_dim = np.ones([3, 1])
    mean_wrong_type = 0.

    with pytest.raises(ValueError):
        IsotropicGaussianMeasure(mean=np.ones(3), variance=wrong_variance)

    with pytest.raises(TypeError):
        IsotropicGaussianMeasure(mean=mean_wrong_type, variance=1.)

    with pytest.raises(ValueError):
        IsotropicGaussianMeasure(mean=mean_wrong_dim, variance=1.)
