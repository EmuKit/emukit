# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ..core.interfaces.models import IModel

try:
    import pyrfr.regression as reg
except ImportError:
    raise ImportError("""
        This module is missing required dependencies. Try running

        pip install git+https://github.com/automl/random_forest_run.git
    """)


class RandomForest(IModel):

    def __init__(self, X_init: np.ndarray, Y_init: np.ndarray, num_trees: int = 30,
                 do_bootstrapping: bool = True, n_points_per_tree: int = 0, seed: int = None) -> None:
        """
        Interface to random forests for Bayesian optimization based on pyrfr package which due to the random splitting
        gives better uncertainty estimates than the sklearn random forest.

        Dependencies:
            AutoML rfr (https://github.com/automl/random_forest_run)

        :param X_init: Initial input data points to train the model
        :param Y_init: Initial target values
        :param num_trees: Specifies the number of trees to build the random forest
        :param do_bootstrapping: Defines if we use boostrapping for the individual trees or not
        :param n_points_per_tree: Specifies the number of points for each individual tree (0 mean no restriction)
        :param seed: Used to seed the random number generator for the random forest (None means random seed)
        """
        super().__init__()

        # Set random number generator for the random forest
        if seed is None:
            seed = np.random.randint(10000)
        self.reg_rng = reg.default_random_engine(seed)

        self.n_points_per_tree = n_points_per_tree

        self.rf = reg.binary_rss_forest()
        self.rf.options.num_trees = num_trees

        self.rf.options.do_bootstrapping = do_bootstrapping

        self.rf.options.num_data_points_per_tree = n_points_per_tree

        self._X = X_init
        self._Y = Y_init

        if self.n_points_per_tree == 0:
            self.rf.options.num_data_points_per_tree = X_init.shape[0]

        data = reg.default_data_container(self._X.shape[1])

        for row_X, row_y in zip(X_init, Y_init):
            data.add_data_point(row_X, row_y)

        self.rf.fit(data, self.reg_rng)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for given points

        :param X: points to run prediction for
        """
        mean = np.zeros([X.shape[0], 1])
        var = np.zeros([X.shape[0], 1])

        for i, x in enumerate(X):
            mean[i], var[i] = self.rf.predict_mean_var(x)

        return mean, var

    def update_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Updates model with new data points.

        :param X: new points
        :param Y: function values at new points X
        """
        self._X = X
        self._Y = Y

        data = reg.default_data_container(self.X.shape[1])

        for row_X, row_y in zip(self._X, self._Y):
            data.add_data_point(row_X, row_y)

        self.rf.fit(data, self.reg_rng)

    def optimize(self) -> None:
        pass

    def get_f_minimum(self):
        """Return the minimum value that has been observed"""
        return np.min(self._Y)

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y
