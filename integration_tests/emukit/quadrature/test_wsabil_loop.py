import GPy
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest_lazyfixture import lazy_fixture

from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, QuadratureRBFGaussianMeasure, RBFGPy
from emukit.quadrature.loop import WSABILLoop
from emukit.quadrature.measures import GaussianMeasure
from emukit.quadrature.methods import WSABIL


def func(x):
    return np.ones((x.shape[0], 1))


@pytest.fixture
def base_gp_data():
    X = np.array([[-1, 1], [0, 0], [-2, 0.1]])
    Y = np.array([[1], [2], [3]])
    gpy_model = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(input_dim=X.shape[1]))
    measure = GaussianMeasure(mean=np.array([0.1, 1.8]), variance=0.8)
    qrbf = QuadratureRBFGaussianMeasure(RBFGPy(gpy_model.kern), measure=measure)
    base_gp = BaseGaussianProcessGPy(kern=qrbf, gpy_model=gpy_model)
    return base_gp, X, Y


@pytest.fixture
def wsabil_adapt(base_gp_data):
    base_gp, X, Y = base_gp_data
    wsabil = WSABIL(base_gp=base_gp, X=X, Y=Y, adapt_alpha=True)
    return wsabil, X, Y


@pytest.fixture
def wsabil_fixed(base_gp_data):
    base_gp, X, Y = base_gp_data
    wsabil = WSABIL(base_gp=base_gp, X=base_gp.X, Y=base_gp.Y, adapt_alpha=True)
    return wsabil, X, Y


@pytest.fixture
def loop_adapt(wsabil_adapt):
    emukit_model, X, Y = wsabil_adapt
    emukit_loop = WSABILLoop(model=emukit_model)
    return emukit_loop, Y.shape[0], X, Y


@pytest.fixture
def loop_fixed(wsabil_fixed):
    emukit_model, X, Y = wsabil_fixed
    emukit_loop = WSABILLoop(model=emukit_model)
    return emukit_loop, Y.shape[0], X, Y


wsabi_test_list = [lazy_fixture("loop_adapt"), lazy_fixture("loop_fixed")]


@pytest.mark.parametrize("loop", wsabi_test_list)
def test_wsabil_loop(loop):
    emukit_loop, init_size, _, _ = loop
    num_iter = 5

    emukit_loop.run_loop(user_function=UserFunctionWrapper(func), stopping_condition=num_iter)

    assert emukit_loop.loop_state.X.shape[0] == num_iter + init_size
    assert emukit_loop.loop_state.Y.shape[0] == num_iter + init_size


@pytest.mark.parametrize("loop", wsabi_test_list)
def test_wsabil_loop_initial_state(loop):
    emukit_loop, _, x_init, y_init = loop

    assert_array_equal(emukit_loop.loop_state.X, x_init)
    assert_array_equal(emukit_loop.loop_state.Y, y_init)
    assert emukit_loop.loop_state.iteration == 0
