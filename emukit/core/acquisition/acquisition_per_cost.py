from typing import Tuple

import numpy as np

from ..acquisition import Acquisition
from ..interfaces import IDifferentiable, IModel


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
        """
        :param cost_model: Model of cost. Should return only positive predictions
        :param min_cost: A minimum value for the cost. The cost model prediction will be clipped to this value if
                         required
        """
        self.cost_model = cost_model
        self.min_cost = min_cost

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate acquisition function

        :param x: A numpy array of shape (n_inputs x n_input_features) containing input locations at which to evaluate
                  the cost
        :return: Value of expected cost at input locations
        """
        return np.maximum(self.cost_model.predict(x)[0], self.min_cost)

    @property
    def has_gradients(self) -> bool:
        """
        Whether gradients of the cost function with respect to input location are available
        """
        return isinstance(self.cost_model, IDifferentiable)

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param x: A numpy array of shape (n_inputs x n_input_features) containing input locations at which to evaluate
                  the cost
        :return: Tuple of numpy arrays: (cost value, cost gradients)
        """
        expected_cost = self.cost_model.predict(x)[0]
        is_below_min_cost = (expected_cost < self.min_cost).flatten()
        grad = self.cost_model.get_prediction_gradients(x)[0]
        grad[is_below_min_cost, :] = 0
        return np.maximum(expected_cost, self.min_cost), grad
