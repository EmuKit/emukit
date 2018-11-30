import numpy as np
import pytest
import pytest_lazyfixture
from scipy.optimize import check_grad

from bayesian_optimization.test_entropy_search import entropy_search_acquisition
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement, NegativeLowerConfidenceBound
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.bayesian_optimization.acquisitions.log_acquisition import LogAcquisition
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.acquisition.acquisition_per_cost import CostAcquisition

from emukit.bayesian_optimization.acquisitions import ProbabilityOfImprovement
from emukit.experimental_design.model_based.acquisitions import ModelVariance, IntegratedVarianceReduction
from emukit.quadrature.acquisitions import SquaredCorrelation

default_grad_tol = 1e-7
gradient_acquisitions = [('negative_lower_confidence_bound_acquisition', default_grad_tol),
                         ('expected_improvement_acquisition', default_grad_tol),
                         ('cost_acquisition', default_grad_tol),
                         ('log_acquisition', 1e-5),
                         ('probability_of_improvement_acquisition', default_grad_tol),
                         ('model_variance_acquisition', 1e-5),
                         ('squared_correlation_acquisition', 1e-3)]

non_gradient_acquisitions = ['entropy_search_acquisition',
                             'multi_source_entropy_search_acquisition',
                             'integrated_variance_acquisition']


# Acquisition function fixtures
@pytest.fixture
def negative_lower_confidence_bound_acquisition(gpy_model):
    return NegativeLowerConfidenceBound(gpy_model)


@pytest.fixture
def expected_improvement_acquisition(gpy_model):
    return ExpectedImprovement(gpy_model)


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
def create_all_fixtures():
    gradient_acquisitions_names = [tup[0] for tup in gradient_acquisitions]
    acquisition_names = gradient_acquisitions_names + non_gradient_acquisitions
    return [pytest.param(pytest_lazyfixture.lazy_fixture(name), id=name) for name in acquisition_names]


def create_gradient_acquisition_fixtures():
    # Create list of tuples of parameters with (fixture, tolerance)
    return [pytest.param(pytest_lazyfixture.lazy_fixture(acq[0]), acq[1], id=acq[0]) for acq in gradient_acquisitions]


# Tests
@pytest.mark.parametrize('acquisition', create_all_fixtures())
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
        err = check_grad(acq, grad, xi, epsilon=1e-8)
        assert err < tol


@pytest.mark.parametrize(('acquisition', 'tol'), create_gradient_acquisition_fixtures())
def test_acquisition_gradient_shapes(acquisition, n_dims, tol):
    rng = np.random.RandomState(43)
    x_test = rng.rand(10, n_dims)

    gradients = acquisition.evaluate_with_gradients(x_test)[1]
    assert gradients.shape == (10, n_dims)
