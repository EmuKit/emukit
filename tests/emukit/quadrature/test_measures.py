# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from utils import check_grad

from emukit.quadrature.measures import GaussianMeasure, UniformMeasure

REL_TOL = 1e-5
ABS_TOL = 1e-4


@dataclass
class DataUniformMeasure:
    D = 2
    measure_bounds = [(-1, 2), (0, 1)]
    measure = UniformMeasure(bounds=measure_bounds)
    dat_bounds = measure_bounds


@dataclass
class DataGaussIsoMeasure:
    D = 2
    mean = np.array([0, 0.8])
    variance = 0.6
    measure = GaussianMeasure(mean=mean, variance=variance)
    dat_bounds = [(m - 2 * np.sqrt(0.5), m + 2 * np.sqrt(0.5)) for m in mean]


@dataclass
class DataGaussMeasure:
    D = 2
    mean = np.array([0, 0.8])
    variance = np.array([0.2, 1.4])
    measure = GaussianMeasure(mean=mean, variance=variance)
    dat_bounds = [(m - 2 * np.sqrt(v), m + 2 * np.sqrt(v)) for m, v in zip(mean, variance)]


@pytest.fixture()
def uniform_measure():
    return DataUniformMeasure()


@pytest.fixture()
def gauss_iso_measure():
    return DataGaussIsoMeasure()


@pytest.fixture()
def gauss_measure():
    return DataGaussMeasure()


measure_test_list = [
    lazy_fixture("uniform_measure"),
    lazy_fixture("gauss_iso_measure"),
    lazy_fixture("gauss_measure"),
]


# === tests shared by all measures start here


@pytest.mark.parametrize("measure", measure_test_list)
def test_measure_gradient_values(measure):
    D, measure, dat_bounds = measure.D, measure.measure, measure.dat_bounds
    func = lambda x: measure.compute_density(x)
    dfunc = lambda x: measure.compute_density_gradient(x).T
    check_grad(func, dfunc, in_shape=(3, D), bounds=dat_bounds)


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


# == tests specific to gaussian measure start here


def test_gauss_measure_values(gauss_measure, gauss_iso_measure):
    assert gauss_iso_measure.is_isotropic
    assert not gauss_measure.is_isotropic


@pytest.mark.parametrize(
    "wrong_input",
    [
        (np.ones(3), -2.0),
        (np.ones(3), -np.array([0.0, 1.0, 2.0])),
        (np.ones(3), -np.array([1.0, -1.0, 2.0])),
        (np.ones(3), -np.array([1.0, 2.0])),
        (np.ones(3), -np.array([1.0, -1.0, 2.0])[:, None]),
    ],
)
def test_gauss_measure_invalid_variance_raises(wrong_input):
    mean, var_wrong_value = wrong_input
    with pytest.raises(ValueError):
        GaussianMeasure(mean=mean, variance=var_wrong_value)


def test_gauss_measure_invalid_mean_raises():
    mean_wrong_shape = np.ones([3, 1])
    with pytest.raises(ValueError):
        GaussianMeasure(mean=mean_wrong_shape, variance=1.0)

    mean_wrong_type = 0.0
    with pytest.raises(TypeError):
        GaussianMeasure(mean=mean_wrong_type, variance=1.0)
