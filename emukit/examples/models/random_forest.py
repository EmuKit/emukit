# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ...core.interfaces.models import IModel

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    raise ImportError("""
        This module is missing required dependencies. Try running:

        pip install scikit-learn

        Refer to http://scikit-learn.org/stable/install.html for further information.
    """)


class RandomForest(IModel):

    def __init__(self, X_init: np.ndarray, Y_init: np.ndarray, num_trees: int = 30,
                 do_bootstrapping: bool = True, seed: int = None) -> None:
        """
        Interface to random forests for Bayesian optimization based on Scikit Learn package.

        Dependencies:
            Scikit Learn (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

        :param X_init: Initial input data points to train the model
        :param Y_init: Initial target values
        :param num_trees: Specifies the number of trees to build the random forest
        :param do_bootstrapping: Defines if we use boostrapping for the individual trees or not
        :param seed: Used to seed the random number generator for the random forest (None means random seed)
        """
        super().__init__()

        self._X = X_init
        self._Y = Y_init

        self.rf = RandomForestRegressor(n_estimators=num_trees, bootstrap=do_bootstrapping, random_state=seed)
        self.rf.fit(X_init, Y_init[:, 0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for given points

        :param X: points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """

        preds = []
        for estimator in self.rf.estimators_:
            pred = estimator.predict(X)
            preds.append(pred)
        mean = np.array(preds).mean(axis=0)[:, None]
        var = np.array(preds).var(axis=0)[:, None]
        return mean, var

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model

        :param X: new points
        :param Y: function values at new points X
        """
        self._X = X
        self._Y = Y

        self.rf.fit(X, Y[:, 0])

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
