# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import numpy as np
import pytest
from utils import check_grad

from emukit.quadrature.measures import BoxDomain, GaussianMeasure, LebesgueMeasure

REL_TOL = 1e-5
ABS_TOL = 1e-4


@dataclass
class DataLebesgueMeasure:
    D = 2
    bounds = [(-1, 2), (0, 1)]
    volume = 3
    measure = LebesgueMeasure(domain=BoxDomain(bounds=bounds), normalized=False)
    dat_bounds = bounds


@dataclass
class DataLebesgueNormalizedMeasure:
    D = 2
    bounds = [(-1, 2), (0, 1)]
    volume = 3
    measure = LebesgueMeasure(domain=BoxDomain(bounds=bounds), normalized=True)
    dat_bounds = bounds


@dataclass
class DataGaussIsoMeasure:
    D = 2
    mean = np.array([0, 0.8])
    variance = 0.6
    measure = GaussianMeasure(mean=mean, variance=variance)
    dat_bounds = [(m - 2 * np.sqrt(0.6), m + 2 * np.sqrt(0.6)) for m in mean]
    reasonable_box_bounds = [(m - 10 * np.sqrt(0.6), m + 10 * np.sqrt(0.6)) for m in mean]


@dataclass
class DataGaussMeasure:
    D = 2
    mean = np.array([0, 0.8])
    variance = np.array([0.2, 1.4])
    measure = GaussianMeasure(mean=mean, variance=variance)
    dat_bounds = [(m - 2 * np.sqrt(v), m + 2 * np.sqrt(v)) for m, v in zip(mean, variance)]
    reasonable_box_bounds = [(m - 10 * np.sqrt(v), m + 10 * np.sqrt(v)) for m, v in zip(mean, variance)]


@pytest.fixture()
def lebesgue_measure():
    return DataLebesgueMeasure()


@pytest.fixture()
def lebesgue_measure_normalized():
    return DataLebesgueNormalizedMeasure()


@pytest.fixture()
def gauss_iso_measure():
    return DataGaussIsoMeasure()


@pytest.fixture()
def gauss_measure():
    return DataGaussMeasure()


measure_test_list = [
    "lebesgue_measure",
    "lebesgue_measure_normalized",
    "gauss_iso_measure",
    "gauss_measure"
]


# === tests shared by all measures start here


@pytest.mark.parametrize("measure_name", measure_test_list)
def test_measure_gradient_values(measure_name, request):
    measure = request.getfixturevalue(measure_name)
    D, measure, dat_bounds = measure.D, measure.measure, measure.dat_bounds
    func = lambda x: measure.compute_density(x)
    dfunc = lambda x: measure.compute_density_gradient(x).T
    check_grad(func, dfunc, in_shape=(3, D), bounds=dat_bounds)


@pytest.mark.parametrize("measure_name", measure_test_list)
def test_measure_shapes(measure_name, request):
    measure = request.getfixturevalue(measure_name)
    D, measure = measure.D, measure.measure

    # box bounds
    bounds = measure.reasonable_box()
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
    res = measure.sample(N)
    assert res.shape == (N, D)


# == tests specific to Lebesgue measure start here


def test_lebesgue_measure_values(lebesgue_measure, lebesgue_measure_normalized):
    from math import isclose

    m = lebesgue_measure.measure
    dat = lebesgue_measure

    m_norm = lebesgue_measure_normalized.measure
    dat_norm = lebesgue_measure_normalized

    assert not m.is_normalized
    assert m_norm.is_normalized

    assert m.can_sample
    assert m_norm.can_sample

    assert m.input_dim == dat.D
    assert m_norm.input_dim == dat_norm.D

    assert m.density == 1.0
    assert isclose(m_norm.density, 1 / dat_norm.volume)

    assert m.reasonable_box() == dat.bounds
    assert m_norm.reasonable_box() == dat_norm.bounds


def test_lebesgue_measure_raises():
    # upper bound smaller than lower bound
    wrong_bounds = [(-1, 1), (0, -2)]
    with pytest.raises(ValueError):
        LebesgueMeasure.from_bounds(bounds=wrong_bounds)

    # empty domain
    wrong_bounds = []
    with pytest.raises(ValueError):
        LebesgueMeasure.from_bounds(bounds=wrong_bounds)


# == tests specific to gaussian measure start here


def test_gauss_measure_values(gauss_measure, gauss_iso_measure):
    m = gauss_measure.measure
    dat = gauss_measure

    m_iso = gauss_iso_measure.measure
    dat_iso = gauss_iso_measure

    assert m_iso.is_isotropic
    assert not m.is_isotropic

    assert m_iso.input_dim == dat_iso.D
    assert not m.is_isotropic == dat.D

    assert all(m_iso.mean == dat_iso.mean)
    assert all(gauss_measure.measure.mean == dat.mean)

    assert all(m_iso.variance == dat_iso.variance)
    assert all(gauss_measure.measure.variance == dat.variance)

    assert np.allclose(m.reasonable_box(), dat.reasonable_box_bounds)
    assert np.allclose(m_iso.reasonable_box(), dat_iso.reasonable_box_bounds)


def test_gauss_measure_shapes(gauss_measure, gauss_iso_measure):
    input_dim = gauss_measure.D
    measure = gauss_measure.measure
    assert measure.mean.shape == (input_dim,)
    assert measure.variance.shape == (input_dim,)
    assert measure.full_covariance_matrix.shape == (input_dim, input_dim)


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
