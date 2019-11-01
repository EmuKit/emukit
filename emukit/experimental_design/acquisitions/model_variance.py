# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

import numpy as np

from ...core.acquisition import Acquisition
from ...core.interfaces import IModel, IDifferentiable


class ModelVariance(Acquisition):
    """
    This acquisition selects the point in the domain where the predictive variance is the highest
    """
    def __init__(self, model: Union[IModel, IDifferentiable]) -> None:
        self.model = model

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        _, variance = self.model.predict(x)
        return variance

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        _, variance = self.model.predict(x)
        _, dvariance_dx = self.model.get_prediction_gradients(x)
        return variance, dvariance_dx

    @property
    def has_gradients(self):
        return isinstance(self.model, IDifferentiable)
