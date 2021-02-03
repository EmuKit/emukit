from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound
from emukit.core.interfaces import IModel, IDifferentiable
import numpy as np
from typing import Union

class DynamicNegativeLowerConfidenceBound(NegativeLowerConfidenceBound):

    def __init__(self, model: Union[IModel, IDifferentiable], input_space_size: int, delta: float) -> None:
        """
        Dynamic extension of the LCB acquisition. The beta coefficient is updated at each iteration, based on the explorativeness parameter delta which is inversely
        proportional to beta itself - the higher the delta the less explorative the selection will be.
        Please consider that regret bounds based on the dynamic exploration coefficient only hold for selected kernel classes exhibiting boundedness and smoothness.
        See the base class for paper references.
        This class may also be taken as a reference for acquisition functions that dynamically update their parameters thanks to the update_parameters() hook; the implicit assumption is that this method is invoked once per iteration (it is no big deal if this is the case for a constant number of times per iteration; should it be more then we are increasing the beta too fast).
        :param model: The underlying model that provides the predictive mean and variance for the given test points
        :param input_space_size: the size of the finite D grid on which the function is evaluated
        :param delta: the exploration parameter determining the beta exploration coefficient; delta must be in (0, 1) and it is inversely related to beta
        """
        assert input_space_size > 0, "Invalid dimension provided"
        assert 0 < delta < 1, "Delta must be in (0, 1)"
        super().__init__(model)
        self.input_space_size = input_space_size
        self.delta = delta
        self.iteration = 0

    def optimal_beta_selection(self) -> float:
        return 2 * np.log(self.input_space_size * (self.iteration ** 2) * (np.pi ** 2) / (6 * self.delta))

    def update_parameters(self) -> None:
        self.iteration += 1
        self.beta = self.optimal_beta_selection()
