import abc

import numpy as np

from .loop_state import LoopState
from ..acquisition import Acquisition
from ..optimization import AcquisitionOptimizer


class CandidatePointCalculator(abc.ABC):
    """ Computes the next point(s) for function evaluation """
    @abc.abstractmethod
    def compute_next_points(self, loop_state: LoopState) -> np.ndarray:
        """
        :param loop_state: Object that contains current state of the loop
        :return: (n_points x n_dims) array of next inputs to evaluate the function at
        """
        pass


class Sequential(CandidatePointCalculator):
    """ This candidate point calculator chooses one candidate point at a time """
    def __init__(self, acquisition: Acquisition, acquisition_optimizer: AcquisitionOptimizer) -> None:
        """
        :param acquisition: Acquisition function to maximise
        :param acquisition_optimizer: Optimizer of acquisition function
        """
        self.acquisition = acquisition
        self.acquisition_optimizer = acquisition_optimizer

    def compute_next_points(self, loop_state: LoopState) -> np.ndarray:
        """
        Computes point(s) to evaluate next

        :param loop_state: Object that contains current state of the loop
        :return: List of function inputs to evaluate the function at next
        """
        x, _ = self.acquisition_optimizer.optimize(self.acquisition)
        return np.atleast_2d(x)
