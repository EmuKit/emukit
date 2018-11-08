from typing import Tuple

import numpy as np

from ...core.acquisition import Acquisition
from ...core.interfaces import IDifferentiable, IModel


def acquisition_per_expected_cost(acquisition: Acquisition, cost_model: IModel, min_cost: float=1e-4) -> Acquisition:
    """
    Creates an acquisition function that is the original acquisition scaled by the expected value of the evaluation
    cost of the user function.

    :param acquisition: Base acquisition function
    :param cost_model: Model of the evaluation cost. Should return positive values only.
    :return: Scaled acquisition function
    """
    return acquisition / CostAcquisition(cost_model, min_cost)


class CostAcquisition(Acquisition):
    """
    Acquisition that simply returns the expected value from the cost model
    """
    def __init__(self, cost_model: IModel, min_cost: float=1e-4):
        self.cost_model = cost_model
        self.min_cost = min_cost

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(self.cost_model.predict(x)[0], self.min_cost)

    @property
    def has_gradients(self) -> bool:
        return isinstance(self.cost_model, IDifferentiable)

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        is_below_min_cost = self.cost_model.predict(x)[0] < self.min_cost
        grad = self.cost_model.get_prediction_gradients(x)[0]
        grad[is_below_min_cost, :] = 0
        return grad
