import GPy
import numpy as np
from scipy.optimize import check_grad

from emukit.bayesian_optimization.acquisitions import MultipointExpectedImprovement
from emukit.model_wrappers import GPyModelWrapper

# Tolerance needs to be quite high since the q-EI is also an approximation.
TOL = 5e-3
# Tolerance for the gradient can be smaller since the approximation is not stochastic
TOL_GRAD = 1e-6
# Tolerance for the gradient of the fast method needs to be higher as it is an approximation of an approximation
TOL_GRAD_FAST = 1e-4


def test_acquisition_multipoint_expected_improvement():
    '''
    Check if the q-EI acquisition function produces similar results as sampling
    '''
    # Batch size
    k = 2

    # Set mean to one
    mu = np.ones((k))

    # Sample random 3 dimensional covarinace matrix:
    L = np.tril(np.random.sample((k, k)))
    Sigma = L @ L.T

    # Set current minimum to a random number smaller than the mean:
    current_minimum = np.random.uniform()

    # Compute acquisition:
    qei_analytic, _, _ = MultipointExpectedImprovement(None)._get_acquisition(mu, Sigma, current_minimum)
    acq_fast = MultipointExpectedImprovement(None, fast_compute=True, eps=1e-3)
    qei_analytic_fast, _, _ = acq_fast._get_acquisition(mu, Sigma, current_minimum)

    # Reference with sampling
    N = 1000000
    samples = np.random.multivariate_normal(mu, Sigma, size=N)
    qei_sampled = current_minimum - np.min(samples, axis=1)
    qei_sampled = sum(qei_sampled[qei_sampled > 0]) / float(N)

    assert np.abs(qei_sampled - qei_analytic) < TOL
    assert np.abs(qei_analytic_fast - qei_analytic) < TOL

def test_acquisition_gradient_multipoint_expected_improvement():
    '''
    Check the q-EI acquisition function gradients with numeric differentiation
    '''
    x_init = np.random.rand(3, 1)
    y_init = np.random.rand(3, 1)
    # Make GPy model
    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = GPyModelWrapper(gpy_model)

    x0 = np.array([0.45, 0.55])
    _check_grad(MultipointExpectedImprovement(model), TOL_GRAD, x0)
    _check_grad(MultipointExpectedImprovement(model, fast_compute=True, eps=1e-3), TOL_GRAD_FAST, x0)

def _check_grad(lp, tol, x0):
    grad_error = check_grad(lambda x: lp.evaluate(x[:, None]).flatten(),
                            lambda x: lp.evaluate_with_gradients(x[:, None])[1].flatten(), x0)

    assert np.all(grad_error < tol)
