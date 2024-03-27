# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import GPy
import numpy as np
import pytest

import emukit.multi_fidelity
import emukit.test_functions.forrester
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel


class TestModels:
    @pytest.fixture()
    def functions(self):
        return [
            lambda x: emukit.test_functions.forrester.forrester_low(x, 0),
            lambda x: emukit.test_functions.forrester.forrester(x, 0),
        ]

    @pytest.fixture()
    def x_init(self, functions):
        np.random.seed(123)
        n_fidelities = len(functions)
        x_init = np.zeros((5 * n_fidelities, 2))
        for i in range(n_fidelities):
            x_init[i * 5 : (i + 1) * 5, 0] = np.random.rand(5)
            x_init[i * 5 : (i + 1) * 5, 1] = i
        return x_init

    @pytest.fixture()
    def y_init(self, x_init, functions):
        n_fidelities = len(functions)
        y_init = np.zeros((5 * n_fidelities, 1))
        for i in range(n_fidelities):
            is_this_fidelity = x_init[:, -1] == i
            y_init[is_this_fidelity, :] = functions[i](x_init[is_this_fidelity, :-1])
        return y_init

    @pytest.fixture
    def model(self, x_init, y_init, functions):
        n_fidelities = len(functions)
        base_kernels = [GPy.kern.RBF(1) for _ in range(len(functions))]
        k = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(base_kernels)

        # Train model
        np.random.seed(123)
        gpy_model = GPyLinearMultiFidelityModel(x_init, y_init, k, n_fidelities)
        model = GPyMultiOutputWrapper(gpy_model, n_fidelities, n_optimization_restarts=5)
        model.optimize()
        return model

    def test_linear_model(self, model, x_init, y_init):
        """
        Make the linear model and optimize it
        """

        # Check predictions are close to true value
        mean_square_error = np.mean(np.square(model.predict(x_init)[0] - y_init))

        assert mean_square_error < 1e-2

    def test_gradients(self, model):
        """
        Ensure model gradients are correct
        """
        model.gpy_model.mixed_noise.Gaussian_noise.fix(1e-2)
        model.gpy_model.mixed_noise.Gaussian_noise_1.fix(1e-2)
        model.optimize()

        # Calculate analytical gradients
        x_test = np.random.rand(1, 2)
        x_test[:, 1] = 1
        dmean_dx, dvar_dx = model.get_prediction_gradients(x_test)

        # Calculate numerical gradients
        eps = 1e-6
        mean, var = model.predict(x_test)

        x_test_dx = x_test.copy()
        x_test_dx[:, 0] += eps
        d_mean, d_var = model.predict(x_test_dx)
        numerical_mean_gradient = (d_mean - mean) / eps
        numerical_var_gradient = (d_var - var) / eps

        # Check gradients are correct
        assert np.abs(numerical_mean_gradient - dmean_dx) < 1e-2
        assert np.abs(numerical_var_gradient - dvar_dx) < 1e-2

    def test_update_data(self, model, functions):
        """
        Check updating model correctly sets new X/Y values
        """

        new_x = np.array([[0.5, 0], [0.5, 1]])
        new_y = np.array([[0.5], [0.5]])

        model.set_data(new_x, new_y)

        assert model.gpy_model.X.shape[0] == 2
        assert model.gpy_model.Y.shape[0] == 2

    def test_calculate_variance_reduction(self, model, functions):
        """
        Compare the analytical comparison of variance reduction to the reduction we get by explicitly adding another
        training point into the model
        """
        x_test_high = np.array([[0.24, 1.0]])
        x_test_low = np.array([[0.24, 0.0]])

        model.gpy_model.mixed_noise.Gaussian_noise.variance.fix(1e-3)
        model.gpy_model.mixed_noise.Gaussian_noise_1.variance.fix(1e-3)
        var_reduction = model.calculate_variance_reduction(x_test_low, x_test_high)

        X_new = np.concatenate([model.gpy_model.X, x_test_low], axis=0)
        y_new = np.concatenate([model.gpy_model.Y, np.array([[0.0]])], axis=0)

        old_var = model.predict(x_test_high)[1]
        model.set_data(X_new, y_new)
        new_var = model.predict(x_test_high)[1]

        var_diff = old_var - new_var
        assert np.isclose(var_reduction, var_diff)


class TestInvalidInputs:
    """
    Test for failure conditions
    """

    @pytest.fixture()
    def kernel(self):
        """
        Make kernel
        """
        n_fidelities = 2
        base_kernels = [GPy.kern.RBF(1) for _ in range(n_fidelities)]
        return emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(base_kernels)

    def test_array_training_input(self, kernel):
        """
        Test failure if lists rather than arrays are passed to model
        """

        with pytest.raises(ValueError):
            GPyLinearMultiFidelityModel([np.random.rand(5, 1)], [np.random.rand(5, 1)], kernel, 1)

    def test_inconsistent_fidelity_indices(self, kernel):
        """
        Test failure if more fidelity indices than fidelities
        """

        x = np.random.rand(5, 2)
        x[:, -1] = 3
        y = np.random.rand(5, 1)

        with pytest.raises(ValueError):
            GPyLinearMultiFidelityModel(x, y, kernel, 2)

    def test_1d_training_inputs(self, kernel):
        """
        Test failure if 1d training data is given
        """

        x = np.random.rand(5, 2)
        y = np.random.rand(5)

        with pytest.raises(ValueError):
            GPyLinearMultiFidelityModel(x, y, kernel, 2)
