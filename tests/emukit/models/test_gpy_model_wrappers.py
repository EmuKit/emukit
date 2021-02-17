import GPy
import numpy as np
import pytest

from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper


@pytest.fixture
def test_data(gpy_model):
    np.random.seed(42)
    return np.random.randn(5, gpy_model.X.shape[1])


@pytest.fixture
def test_data2(gpy_model):
    np.random.seed(42)
    return np.random.randn(4, gpy_model.X.shape[1])


def test_joint_prediction_gradients(gpy_model, test_data):
    epsilon = 1e-5
    mean, cov = gpy_model.predict_with_full_covariance(test_data)
    # Get the gradients
    mean_dx, cov_dx = gpy_model.get_joint_prediction_gradients(test_data)

    for i in range(test_data.shape[0]):  # Iterate over each test point
        for j in range(test_data.shape[1]):  # Iterate over each dimension
            # Approximate the gradient numerically
            perturbed_input = test_data.copy()
            perturbed_input[i, j] += epsilon
            mean_perturbed, cov_perturbed = gpy_model.predict_with_full_covariance(perturbed_input)
            mean_dx_numerical = (mean_perturbed - mean) / epsilon
            cov_dx_numerical = (cov_perturbed - cov) / epsilon
            # Check that numerical approx. similar to true gradient
            assert pytest.approx(mean_dx_numerical, abs=1e-8, rel=1e-3) == mean_dx[:, :, i, j]
            assert pytest.approx(cov_dx_numerical, abs=1e-8, rel=1e-3) == cov_dx[:, :, i, j]
    

def test_get_covariance_between_points_gradients(gpy_model, test_data, test_data2):
    epsilon = 1e-5
    cov = gpy_model.get_covariance_between_points(test_data, test_data2)
    # Get the gradients
    cov_dx = gpy_model.get_covariance_between_points(test_data, test_data2)

    for i in range(test_data.shape[0]):  # Iterate over each test point
        for j in range(test_data.shape[1]):  # Iterate over each dimension
            # Approximate the gradient numerically
            perturbed_input = test_data.copy()
            perturbed_input[i, j] += epsilon
            cov_perturbed = gpy_model.get_covariance_between_points(perturbed_input, test_data2)
            cov_dx_numerical = (cov_perturbed - cov) / epsilon
            # Check that numerical approx. similar to true gradient
            assert pytest.approx(cov_dx_numerical, abs=1e-8, rel=1e-3) == cov_dx[:, :, i, j]
