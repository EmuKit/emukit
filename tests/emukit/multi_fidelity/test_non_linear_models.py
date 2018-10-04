import pytest

import numpy as np
import GPy
from scipy.optimize import check_grad

import emukit.multi_fidelity.models
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels


class TestNonLinearModel:
    @pytest.fixture()
    def x_init(self):
        x_init = np.zeros((15, 3))
        for i in range(3):
            x_init[i * 5:(i + 1) * 5, :2] = np.random.randn(5, 2)
            x_init[i * 5:(i + 1) * 5, 2] = i
        return x_init

    @pytest.fixture()
    def y_init(self):
        y_init = np.zeros((15, 1))
        for i in range(3):
            y_init[i * 5:(i + 1) * 5, :] = np.random.randn(5, 1)
        return y_init

    @pytest.fixture()
    def non_linear_model(self, x_init, y_init):
        """
        Creates a NonLinearModel instance to use in tests
        """
        np.random.seed(123)
        base_kernel = GPy.kern.RBF
        kernel = make_non_linear_kernels(base_kernel, len(x_init), x_init.shape[1] - 1)
        model = emukit.multi_fidelity.models.NonLinearMultiFidelityModel(x_init, y_init, 3, kernel, n_samples=3)
        return model

    def test_invalid_kernel(self, x_init, y_init):
        """
        Check sensible error is thrown if we pass in a kernel instance rather than class definition
        """
        base_kernel = GPy.kern.RBF(1)
        with pytest.raises(TypeError):
            emukit.multi_fidelity.models.NonLinearMultiFidelityModel(x_init, y_init, base_kernel, n_samples=70)

    def test_invalid_input(self, x_init, y_init):
        """
        Test for sensible error message if we pass arrays rather than lists to constructor
        """
        base_kernel = GPy.kern.RBF
        with pytest.raises(TypeError):
            emukit.multi_fidelity.models.NonLinearMultiFidelityModel([np.random.rand(5, 3)], [np.random.rand(5, 3)], base_kernel,
                                                        n_samples=70)
        with pytest.raises(TypeError):
            emukit.multi_fidelity.models.NonLinearMultiFidelityModel([np.random.rand(5, 3)], np.random.rand(5, 3), base_kernel,
                                                        n_samples=70)

    def test_get_fmin(self, non_linear_model):
        """
        Tests get_fmin returns the correct value
        """
        min_value = non_linear_model.get_f_minimum()
        assert min_value == non_linear_model.models[-1].Y.min()

    def test_optimize(self, non_linear_model):
        """
        Tests the optimization doesn't fail
        """
        non_linear_model.optimize()

    def test_update(self, non_linear_model):
        """
        Tests updating the model works
        """

        x = np.zeros((15, 3))
        for i in range(3):
            x[i * 5:(i + 1) * 5, :2] = np.random.randn(5, 2)
            x[i * 5:(i + 1) * 5, 2] = i

        y = np.zeros((15, 1))
        for i in range(3):
            y[i * 5:(i + 1) * 5, :] = np.random.randn(5, 1)

        non_linear_model.update_data(x, y)

        assert non_linear_model.models[0].X.shape == (5, 2)
        assert non_linear_model.models[1].X.shape == (5, 3)
        assert non_linear_model.models[2].X.shape == (5, 3)

        assert non_linear_model.models[0].Y.shape == (5, 1)
        assert non_linear_model.models[1].Y.shape == (5, 1)
        assert non_linear_model.models[2].Y.shape == (5, 1)

    def test_X(self, non_linear_model):
        assert isinstance(non_linear_model.X, np.ndarray)
        assert non_linear_model.X.ndim == 2
        assert non_linear_model.X.shape == (15, 3)

    def test_Y(self, non_linear_model):
        assert isinstance(non_linear_model.Y, np.ndarray)
        assert non_linear_model.Y.ndim == 2
        assert non_linear_model.Y.shape == (15, 1)

    def test_non_linear_model_with_3_fidelities(self, non_linear_model):
        """
        Test the model prediction doesn't fail and shapes are correct
        """

        x_test = np.random.rand(2, 3)
        x_test[:, -1] = 2
        dmean_dx, dvar_dx = non_linear_model.get_prediction_gradients(x_test)
        assert dmean_dx.shape == (2, 2)
        assert dvar_dx.shape == (2, 2)

    def test_non_linear_model_prediction(self, non_linear_model):
        """
        Test the model prediction doesn't fail and shapes are correct
        """
        X = np.random.rand(2, 3)
        X[:, -1] = 2
        mean, var = non_linear_model.predict(X)
        assert mean.shape == (2, 1)
        assert var.shape == (2, 1)

    def test_non_linear_model_prediction_with_grads(self, non_linear_model):
        """
        Test the model prediction doesn't fail and shapes are correct
        """

        x_test = np.random.rand(2, 3)
        x_test[:, -1] = 2
        dmean_dx, dvar_dx = non_linear_model.get_prediction_gradients(x_test)
        assert dmean_dx.shape == (2, 2)
        assert dvar_dx.shape == (2, 2)

    def test_non_linear_sample_mean_gradient_highest_fidelity(self, non_linear_model):
        np.random.seed(1234)
        x0 = np.random.rand(2)

        assert check_grad(lambda x: np.sum(non_linear_model._predict_samples_with_gradients(x[None, :], 2)[0], axis=0),
                          lambda x: np.sum(non_linear_model._predict_samples_with_gradients(x[None, :], 2)[1], axis=0), x0) < 1e-6

    def test_non_linear_sample_var_gradient_highest_fidelity(self, non_linear_model):
        np.random.seed(1234)
        x0 = np.random.rand(2)

        assert check_grad(lambda x: np.sum(non_linear_model._predict_samples_with_gradients(x[None, :], 2)[2], axis=0),
                          lambda x: np.sum(non_linear_model._predict_samples_with_gradients(x[None, :], 2)[3], axis=0), x0) < 1e-6

    def test_non_linear_sample_mean_gradient_middle_fidelity(self, non_linear_model):
        np.random.seed(1234)
        x0 = np.random.rand(2)

        assert check_grad(lambda x: np.sum(non_linear_model._predict_samples_with_gradients(x[None, :], 1)[0], axis=0),
                          lambda x: np.sum(non_linear_model._predict_samples_with_gradients(x[None, :], 1)[1], axis=0), x0) < 1e-6

    def test_non_linear_sample_var_gradient_middle_fidelity(self, non_linear_model):
        np.random.seed(1234)
        x0 = np.random.rand(2)

        assert check_grad(lambda x: np.sum(non_linear_model._predict_samples_with_gradients(x[None, :], 1)[2], axis=0),
                          lambda x: np.sum(non_linear_model._predict_samples_with_gradients(x[None, :], 1)[3], axis=0), x0) < 1e-6

    def test_non_linear_sample_var_gradient_lowest_fidelity(self, non_linear_model):
        np.random.seed(1234)
        x0 = np.random.rand(2)

        assert np.all(check_grad(lambda x: non_linear_model._predict_samples_with_gradients(x[None, :], 0)[2],
                                 lambda x: non_linear_model._predict_samples_with_gradients(x[None, :], 0)[3], x0) < 1e-6)

    def test_non_linear_sample_mean_gradient_lowest_fidelity(self, non_linear_model):
        np.random.seed(1234)
        x0 = np.random.rand(2)

        assert np.all(check_grad(lambda x: non_linear_model._predict_samples_with_gradients(x[None, :], 0)[0],
                                 lambda x: non_linear_model._predict_samples_with_gradients(x[None, :], 0)[1], x0) < 1e-6)

    def test_non_linear_model_mean_gradient(self, non_linear_model):
        """
        Check the gradient of the mean prediction is correct
        """

        np.random.seed(1234)
        x0 = np.random.rand(2)

        # wrap function so fidelity index doesn't change
        def wrap_func(x):
            x_full = np.concatenate([x[None, :], [[2]]], axis=1)
            return non_linear_model.predict(x_full)[0]

        def wrap_gradients(x):
            x_full = np.concatenate([x[None, :], [[2]]], axis=1)
            return non_linear_model.get_prediction_gradients(x_full)[0]
        assert np.all(check_grad(wrap_func, wrap_gradients, x0) < 1e-6)

    def test_non_linear_model_variance_gradient(self, non_linear_model):
        """
        Check the gradient of the predictive variance is correct
        """

        np.random.seed(1234)
        x0 = np.random.rand(2)

        # wrap function so fidelity index doesn't change
        def wrap_func(x):
            x_full = np.concatenate([x[None, :], [[2]]], axis=1)
            return non_linear_model.predict(x_full)[1]

        def wrap_gradients(x):
            x_full = np.concatenate([x[None, :], [[2]]], axis=1)
            return non_linear_model.get_prediction_gradients(x_full)[1]

        assert np.all(check_grad(wrap_func, wrap_gradients, x0) < 1e-6)


def test_non_linear_kernel_ard():
    """
    Test that the kernels that act on the input space have the correct number of lengthscales when ARD is true
    """
    kernels = make_non_linear_kernels(GPy.kern.RBF, 2, 2, ARD=True)
    assert len(kernels[0].lengthscale) == 2
    assert len(kernels[1].bias_kernel_fidelity2.lengthscale) == 2
    assert len(kernels[1].mul.scale_kernel_fidelity2.lengthscale) == 2