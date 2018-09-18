import pytest
import numpy as np

from emukit.models.random_forest import RandomForest


class TestRandomForest:
    @pytest.fixture()
    def rf_model(self):
        """
        Creates a random forest instance to use in tests
        """
        x_init = np.random.randn(5, 2)
        y_init = np.random.randn(5, 1)

        # reduce the number of steps to a bare minimum to make it faster
        model = RandomForest(x_init, y_init)

        return model

    def test_model_optimize_hyperparameters(self, rf_model):
        """
        Tests the optimization doesn't fail
        """
        rf_model.optimize()

    def test_model_update(self, rf_model):
        """
        Tests updating the model works
        """
        n_points = 20
        n_init = rf_model.Y.shape[0]

        x_new = np.random.randn(n_points, 2)
        y_new = np.random.randn(n_points, 1)
        rf_model.update_data(x_new, y_new)

        assert rf_model.Y.shape[0] == n_points + n_init

    def test_model_predict_check_correct_shapes(self, rf_model):
        """
        Test the model prediction doesn't fail and shapes are correct
        """

        x_test = np.random.rand(2, 2)
        mean, var = rf_model.predict(x_test)

        assert mean.shape == (2, 1)
        assert var.shape == (2, 1)
