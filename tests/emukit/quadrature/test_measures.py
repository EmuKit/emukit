# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from utils import check_grad

from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure, UniformMeasure

REL_TOL = 1e-5
ABS_TOL = 1e-4


@dataclass
class DataUniformMeasure:
    D = 2
    measure_bounds = [(-1, 2), (0, 1)]
    measure = UniformMeasure(bounds=measure_bounds)


@dataclass
class DataGaussIsoMeasure:
    D = 2
    mean = np.arange(D)
    variance = 0.5
    measure = IsotropicGaussianMeasure(mean=mean, variance=variance)


@pytest.fixture()
def uniform_measure():
    return DataUniformMeasure()


@pytest.fixture()
def gauss_iso_measure():
    return DataGaussIsoMeasure()


measure_test_list = [
    lazy_fixture("uniform_measure"),
    lazy_fixture("gauss_iso_measure"),
]


# === tests shared by all measures start here


@pytest.mark.parametrize("measure", measure_test_list)
def test_measure_gradient_values(measure):
    D, measure = measure.D, measure.measure
    func = lambda x: measure.compute_density(x)
    dfunc = lambda x: measure.compute_density_gradient(x).T
    check_grad(func, dfunc, in_shape=(3, D))


@pytest.mark.parametrize("measure", measure_test_list)
def test_measure_shapes(measure):
    D, measure = measure.D, measure.measure

    # box bounds
    bounds = measure.get_box()
    assert len(bounds) == D
    for b in bounds:
        assert len(b) == 2

    # density
    N = 5
    np.random.seed(0)
    x = np.random.randn(N, D)

    res = measure.compute_density(x)
    assert res.shape == (N,)

    # sampling capabilities
    assert measure.can_sample
    res = measure.get_samples(N)
    assert res.shape == (N, D)


# == tests specific to uniform measure start here


def test_uniform_measure_wrong_bounds():
    bounds = [(-1, 1), (3, 2), (1.3, 5.0)]

    with pytest.raises(ValueError):
        UniformMeasure(bounds)


# == tests specific to gauss iso measure start here


def test_iso_gauss_measure_invalid_input():
    var_wrong_value = -2.0
    mean_wrong_shape = np.ones([3, 1])
    mean_wrong_type = 0.0

    with pytest.raises(ValueError):
        IsotropicGaussianMeasure(mean=np.ones(3), variance=var_wrong_value)

    with pytest.raises(TypeError):
        IsotropicGaussianMeasure(mean=mean_wrong_type, variance=1.0)

    with pytest.raises(ValueError):
        IsotropicGaussianMeasure(mean=mean_wrong_shape, variance=1.0)
