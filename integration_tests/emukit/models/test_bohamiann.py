import pytest
import numpy as np

from emukit.models.bohamiann import Bohamiann


class TestBohamiann:
    @pytest.fixture()
    def bnn_model(self):
        """
        Creates a Bohamiann instance to use in tests
        """
        x_init = np.random.randn(5, 2)
        y_init = np.random.randn(5, 1)

        # reduce the number of steps to a bare minimum to make it faster
        model = Bohamiann(x_init, y_init, num_steps=100, num_burn_in_steps=10, keep_every=10)

        return model

    def test_model_optimize_hyperparameters(self, bnn_model):
        """
        Tests the optimization doesn't fail
        """
        bnn_model.optimize()

    def test_model_update(self, bnn_model):
        """
        Tests updating the model works
        """
        n_points = 20
        n_init = bnn_model.Y.shape[0]

        x_new = np.random.randn(n_points, 2)
        y_new = np.random.randn(n_points, 1)
        bnn_model.update_data(x_new, y_new)

        assert bnn_model.Y.shape[0] == n_points + n_init

    def test_model_predict_check_correct_shapes(self, bnn_model):
        """
        Test the model prediction doesn't fail and shapes are correct
        """

        x_test = np.random.rand(2, 2)
        mean, var = bnn_model.predict(x_test)

        assert mean.shape == (2, 1)
        assert var.shape == (2, 1)
