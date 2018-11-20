import GPy
import numpy as np
import scipy.optimize
from emukit.model_wrappers import GPyModelWrapper

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.acquisitions.log_acquisition import LogAcquisition


def test_log_acquisition_gradients():
    x_init = np.random.rand(5, 2)
    y_init = np.random.rand(5, 1)

    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = GPyModelWrapper(gpy_model)

    ei = ExpectedImprovement(model)
    log = LogAcquisition(ei)

    x0 = np.random.rand(2)
    assert np.all(scipy.optimize.check_grad(lambda x: log.evaluate_with_gradients(x[None, :])[0],
                                            lambda x: log.evaluate_with_gradients(x[None, :])[1], x0) < 1e-6)


def test_log_acquisition_shapes():
    x_init = np.random.rand(5, 2)
    y_init = np.random.rand(5, 1)

    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = GPyModelWrapper(gpy_model)

    ei = ExpectedImprovement(model)
    log = LogAcquisition(ei)

    x = np.ones((5, 2))
    value, gradient = log.evaluate_with_gradients(x)
    assert value.shape == (5, 1)
    assert gradient.shape == (5, 2)

    value_2 = log.evaluate(x)
    assert value_2.shape == (5, 1)
