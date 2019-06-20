from collections import namedtuple

import numpy as np
import pytest
import pytest_lazyfixture
from scipy.optimize import check_grad

from bayesian_optimization.test_entropy_search import entropy_search_acquisition
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement, NegativeLowerConfidenceBound, EntropySearch
from emukit.core.acquisition import IntegratedHyperParameterAcquisition
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.bayesian_optimization.acquisitions.log_acquisition import LogAcquisition
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.acquisition.acquisition_per_cost import CostAcquisition

from emukit.bayesian_optimization.acquisitions import ProbabilityOfImprovement
from emukit.experimental_design.model_based.acquisitions import ModelVariance, IntegratedVarianceReduction
from emukit.model_wrappers.gpy_quadrature_wrappers import convert_gpy_model_to_emukit_model
from emukit.quadrature.acquisitions import SquaredCorrelation
from emukit.quadrature.methods import VanillaBayesianQuadrature


# This is the step sized used by scipy.optimize.check_grad to calculate the numerical gradient
gradient_check_step_size = 1e-8
default_grad_tol = 1e-7
# rmse_gradient_tolerance is the maximum allowed root mean squared error as calculated by scipy.optimize.check_grad
# before the test will fail
acquisition_test_tuple = namedtuple('AcquisitionTest', ['name', 'has_gradients', 'rmse_gradient_tolerance'])
acquisition_tests = [acquisition_test_tuple('negative_lower_confidence_bound_acquisition', True, default_grad_tol),
                     acquisition_test_tuple('expected_improvement_acquisition', True, default_grad_tol),
                     acquisition_test_tuple('cost_acquisition', True, default_grad_tol),
                     acquisition_test_tuple('log_acquisition', True, 1e-5),
                     acquisition_test_tuple('probability_of_improvement_acquisition', True, default_grad_tol),
                     acquisition_test_tuple('model_variance_acquisition', True, 1e-5),
                     acquisition_test_tuple('squared_correlation_acquisition', True, 1e-3),
                     acquisition_test_tuple('entropy_search_acquisition', False, np.nan),
                     acquisition_test_tuple('multi_source_entropy_search_acquisition', False, np.nan),
                     acquisition_test_tuple('integrated_variance_acquisition', False, np.nan),
                     acquisition_test_tuple('integrated_expected_improvement_acquisition', True, default_grad_tol),
                     acquisition_test_tuple('integrated_probability_of_improvement_acquisition', False, np.nan)]


# Vanilla bq model for squared correlation test
@pytest.fixture
def vanilla_bq_model(gpy_model, continuous_space, n_dims):
    integral_bounds = continuous_space.get_bounds()
    model = convert_gpy_model_to_emukit_model(gpy_model.model, integral_bounds)
    return VanillaBayesianQuadrature(model)


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
def cost_acquisition(gpy_model):
    return CostAcquisition(gpy_model, 1e-6)


@pytest.fixture
def log_acquisition(expected_improvement_acquisition):
    return LogAcquisition(expected_improvement_acquisition)


@pytest.fixture
def probability_of_improvement_acquisition(gpy_model):
    return ProbabilityOfImprovement(gpy_model)


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
@pytest.mark.parametrize('n_dims', [2])
def multi_source_entropy_search_acquisition(gpy_model):
    space = ParameterSpace([ContinuousParameter('x1', 0, 1), InformationSourceParameter(2)])
    return MultiInformationSourceEntropySearch(gpy_model, space, num_representer_points=10)


# Helpers for creating parameterized fixtures
def create_acquisition_fixture_parameters():
    return [pytest.param(pytest_lazyfixture.lazy_fixture(acq.name), id=acq.name) for acq in acquisition_tests]


def create_gradient_acquisition_fixtures():
    # Create list of tuples of parameters with (fixture, tolerance) for acquisitions that gave gradients only
    parameters = []
    for acquisition in acquisition_tests:
        if acquisition.has_gradients:
            acquisition_name = acquisition.name
            lazy_fixture = pytest_lazyfixture.lazy_fixture(acquisition.name)
            parameters.append(pytest.param(lazy_fixture, acquisition.rmse_gradient_tolerance, id=acquisition_name))
    return parameters


# Tests
@pytest.mark.parametrize('acquisition', create_acquisition_fixture_parameters())
def test_acquisition_evaluate_shape(acquisition, n_dims):
    x = np.random.rand(10, n_dims)
    acquisition_value = acquisition.evaluate(x)
    assert acquisition_value.shape == (10, 1)


@pytest.mark.parametrize(('acquisition', 'tol'), create_gradient_acquisition_fixtures())
def test_acquisition_gradient_computation(acquisition, n_dims, tol):
    rng = np.random.RandomState(43)
    x_test = rng.rand(10, n_dims)

    acq = lambda x: acquisition.evaluate(np.array([x]))[0][0]
    grad = lambda x: acquisition.evaluate_with_gradients(np.array([x]))[1][0]

    for xi in x_test:
        err = check_grad(acq, grad, xi, epsilon=gradient_check_step_size)
        assert err < tol


@pytest.mark.parametrize(('acquisition', 'tol'), create_gradient_acquisition_fixtures())
def test_acquisition_gradient_shapes(acquisition, n_dims, tol):
    rng = np.random.RandomState(43)
    x_test = rng.rand(10, n_dims)

    gradients = acquisition.evaluate_with_gradients(x_test)[1]
    assert gradients.shape == (10, n_dims)
