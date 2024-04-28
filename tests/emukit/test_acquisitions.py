# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from collections import namedtuple

import numpy as np
import pytest
from scipy.optimize import check_grad

from emukit.bayesian_optimization.acquisitions import (
    MUMBO,
    EntropySearch,
    ExpectedImprovement,
    MaxValueEntropySearch,
    NegativeLowerConfidenceBound,
    ProbabilityOfFeasibility,
    ProbabilityOfImprovement,
)
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.bayesian_optimization.acquisitions.log_acquisition import LogAcquisition
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace
from emukit.core.acquisition import IntegratedHyperParameterAcquisition
from emukit.core.acquisition.acquisition_per_cost import CostAcquisition
from emukit.experimental_design.acquisitions import IntegratedVarianceReduction, ModelVariance
from emukit.model_wrappers.gpy_quadrature_wrappers import create_emukit_model_from_gpy_model
from emukit.quadrature.acquisitions import MutualInformation, SquaredCorrelation, UncertaintySampling
from emukit.quadrature.methods import VanillaBayesianQuadrature
from emukit.samplers import AffineInvariantEnsembleSampler

# This is the step sized used by scipy.optimize.check_grad to calculate the numerical gradient
gradient_check_step_size = 1e-8
default_grad_tol = 1e-7
# rmse_gradient_tolerance is the maximum allowed root mean squared error as calculated by scipy.optimize.check_grad
# before the test will fail
acquisition_test_tuple = namedtuple("AcquisitionTest", ["name", "has_gradients", "rmse_gradient_tolerance"])
acquisition_tests = [
    acquisition_test_tuple("negative_lower_confidence_bound_acquisition", True, default_grad_tol),
    acquisition_test_tuple("expected_improvement_acquisition", True, default_grad_tol),
    acquisition_test_tuple("cost_acquisition", True, default_grad_tol),
    acquisition_test_tuple("log_acquisition", True, 1e-5),
    acquisition_test_tuple("probability_of_improvement_acquisition", True, default_grad_tol),
    acquisition_test_tuple("model_variance_acquisition", True, 1e-5),
    acquisition_test_tuple("squared_correlation_acquisition", True, 1e-3),
    acquisition_test_tuple("mutual_information_acquisition", True, 1e-3),
    acquisition_test_tuple("uncertainty_sampling_acquisition", True, 1e-3),
    acquisition_test_tuple("entropy_search_acquisition", False, np.nan),
    acquisition_test_tuple("max_value_entropy_search_acquisition", False, np.nan),
    acquisition_test_tuple("multi_source_entropy_search_acquisition", False, np.nan),
    acquisition_test_tuple("MUMBO_acquisition", False, np.nan),
    acquisition_test_tuple("integrated_variance_acquisition", False, np.nan),
    acquisition_test_tuple("integrated_expected_improvement_acquisition", True, default_grad_tol),
    acquisition_test_tuple("integrated_probability_of_improvement_acquisition", False, np.nan),
    acquisition_test_tuple("probability_of_feasibility_acquisition", True, default_grad_tol),
]


# Vanilla bq model used to test bq acquisition functions
@pytest.fixture
def vanilla_bq_model(gpy_model, continuous_space, n_dims):
    integral_bounds = continuous_space.get_bounds()
    model = create_emukit_model_from_gpy_model(gpy_model.model, integral_bounds, None)
    return VanillaBayesianQuadrature(model, model.X, model.Y)


# Acquisition function fixtures
@pytest.fixture
def negative_lower_confidence_bound_acquisition(gpy_model):
    return NegativeLowerConfidenceBound(gpy_model)


@pytest.fixture
def expected_improvement_acquisition(gpy_model):
    return ExpectedImprovement(gpy_model)


@pytest.fixture
def integrated_expected_improvement_acquisition(gpy_model_mcmc):
    return IntegratedHyperParameterAcquisition(gpy_model_mcmc, ExpectedImprovement, 10)


@pytest.fixture
def integrated_probability_of_improvement_acquisition(gpy_model_mcmc):
    return IntegratedHyperParameterAcquisition(gpy_model_mcmc, ProbabilityOfImprovement, 10)


@pytest.fixture
def probability_of_feasibility_acquisition(gpy_model):
    return ProbabilityOfFeasibility(gpy_model)


@pytest.fixture
def cost_acquisition(gpy_model):
    return CostAcquisition(gpy_model, 1e-6)


@pytest.fixture
def log_acquisition(expected_improvement_acquisition):
    return LogAcquisition(expected_improvement_acquisition)


@pytest.fixture
def probability_of_improvement_acquisition(gpy_model):
    return ProbabilityOfImprovement(gpy_model)


@pytest.fixture
def entropy_search_acquisition(gpy_model, continuous_space):
    sampler = AffineInvariantEnsembleSampler(continuous_space)
    return EntropySearch(gpy_model, continuous_space, sampler, num_representer_points=10)


@pytest.fixture
def max_value_entropy_search_acquisition(gpy_model, continuous_space):
    return MaxValueEntropySearch(gpy_model, continuous_space, num_samples=2, grid_size=100)


@pytest.fixture
def model_variance_acquisition(gpy_model):
    return ModelVariance(gpy_model)


@pytest.fixture
def integrated_variance_acquisition(gpy_model, continuous_space):
    return IntegratedVarianceReduction(gpy_model, continuous_space)


@pytest.fixture
def squared_correlation_acquisition(vanilla_bq_model):
    return SquaredCorrelation(vanilla_bq_model)


@pytest.fixture
def mutual_information_acquisition(vanilla_bq_model):
    return MutualInformation(vanilla_bq_model)


@pytest.fixture
def uncertainty_sampling_acquisition(vanilla_bq_model):
    return UncertaintySampling(vanilla_bq_model)


@pytest.fixture
@pytest.mark.parametrize("n_dims", [2])
def multi_source_entropy_search_acquisition(gpy_model):
    space = ParameterSpace([ContinuousParameter("x1", 0, 1), InformationSourceParameter(2)])
    return MultiInformationSourceEntropySearch(gpy_model, space, num_representer_points=10)


@pytest.fixture
@pytest.mark.parametrize("n_dims", [2])
def MUMBO_acquisition(gpy_model):
    space = ParameterSpace([ContinuousParameter("x1", 0, 1), InformationSourceParameter(2)])
    return MUMBO(gpy_model, space, num_samples=10, grid_size=5000)


# Helpers for creating parameterized fixtures
def create_acquisition_fixture_parameters():
    return [acq.name for acq in acquisition_tests]


def create_gradient_acquisition_fixtures():
    # Create list of tuples of parameters with (fixture, tolerance) for acquisitions that gave gradients only
    parameters = []
    for acquisition in acquisition_tests:
        if acquisition.has_gradients:
            parameters.append(pytest.param(acquisition.name, acquisition.rmse_gradient_tolerance, id=acquisition.name))
    return parameters


# Tests
@pytest.mark.parametrize("acquisition_name", create_acquisition_fixture_parameters())
def test_acquisition_evaluate_shape(acquisition_name, n_dims, request):
    acquisition = request.getfixturevalue(acquisition_name)
    x = np.random.rand(1, n_dims)
    acquisition_value = acquisition.evaluate(x)
    assert acquisition_value.shape == (1, 1)

    x = np.random.rand(10, n_dims)
    acquisition_value = acquisition.evaluate(x)
    assert acquisition_value.shape == (10, 1)


@pytest.mark.parametrize(("acquisition_name", "tol"), create_gradient_acquisition_fixtures())
def test_acquisition_gradient_computation(acquisition_name, n_dims, tol, request):
    acquisition = request.getfixturevalue(acquisition_name)
    rng = np.random.RandomState(43)
    x_test = rng.rand(10, n_dims)

    acq = lambda x: acquisition.evaluate(np.array([x]))[0][0]
    grad = lambda x: acquisition.evaluate_with_gradients(np.array([x]))[1][0]

    for xi in x_test:
        err = check_grad(acq, grad, xi, epsilon=gradient_check_step_size)
        assert err < tol


@pytest.mark.parametrize(("acquisition_name", "tol"), create_gradient_acquisition_fixtures())
def test_acquisition_gradient_shapes(acquisition_name, n_dims, tol, request):
    acquisition = request.getfixturevalue(acquisition_name)
    rng = np.random.RandomState(43)

    x_test = rng.rand(1, n_dims)
    gradients = acquisition.evaluate_with_gradients(x_test)[1]
    assert gradients.shape == (1, n_dims)

    x_test = rng.rand(10, n_dims)
    gradients = acquisition.evaluate_with_gradients(x_test)[1]
    assert gradients.shape == (10, n_dims)
