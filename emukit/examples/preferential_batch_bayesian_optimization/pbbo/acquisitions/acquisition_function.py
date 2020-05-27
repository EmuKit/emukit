from typing import Tuple, Dict, Callable, Optional
import numpy as np
import emukit


from .. import ComparisonGP, ComparisonGPEmukitWrapper
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core import ParameterSpace
from emukit.core.optimization import  ContextManager

class AcquisitionFunction():
    """
    This class implements the general AcquisitionFunction
    
    :param options: Dictionary containing the acquisition function options
    :param optimizer_options: Dictionary containing options for the acquisition function optimizer
    """
    def __init__(self, options: Dict={}, optimizer_options: Dict={}):
        self.optimizer_options = optimizer_options
        self.pool_size = options.get('pool_size', -1)
        self.acq_samples = options.get('acq_samples', 1000)
        self.acq_opt_restarts = options.get('acq_opt_restarts', 10)

    def acq_fun_optimizer(self, m: ComparisonGP, bounds: np.ndarray, batch_size: int, get_logger: Callable) -> np.ndarray:
        """
        Implements the optimization scheme for the acquisition function
        
        :param m: The model which posterior is used by the acquisition function (from which the samples are drawn from)
        :param bounds: the optimization bounds of the new sample
        :param batch_size: How many points are there in the batch
        :param get_logger: Function for receiving the legger where the prints are forwarded.
        :return: optimized locations
        """
        raise NotImplementedError
    
    def reset(self, model: ComparisonGPEmukitWrapper) -> None:
        """
        Some acquisition functions need to be reseted, this method is for that.
        :param model: the model to be passed to the acquisition function (some acquisition functions need a model at this point)
        """
        None


class EmukitAcquisitionFunctionWrapper(Acquisition):
    def __init__(self, model: ComparisonGP, acquisitionFunction: AcquisitionFunction) -> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. For more information see:
        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization
        :param model: model that is used to compute the improvement.
        :param acquisitionFunction: The used acquisitionFunction
        """

        self.model = model
        self.acquisitionFunction = acquisitionFunction
        self.size = self.model.X.shape[0]
        
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.
        :param x: points where the acquisition is evaluated.
        :return: acquisition function values
        """
        return -self.acquisitionFunction.evaluate(x, self.model)

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Expected Improvement.
        :param x: points where the acquisition is evaluated.
        :return: a tuple containing the acquisition function values and their gradients
        """
        f,g = self.acquisitionFunction.evaluate_with_gradients(x, self.model)
        return -f, -g
    
    def reset(self, model: ComparisonGPEmukitWrapper) -> None:
        """
        Some acquisition functions need to be reseted, this method is for that.
        :param model: the model to be passed to the acquisition function (some acquisition functions need a model at this point)
        """
        self.acquisitionFunction.reset(model.model)
    
    @property
    def has_gradients(self) -> bool:
        """
        :return: True as all acquisition functions have gradients
        """
        return True

class SequentialGradientAcquisitionOptimizer(GradientAcquisitionOptimizer):

    def __init__(self, space: ParameterSpace, batch_size: int) -> None:
        """
        :param space: The parameter space spanning the search problem.
        """
        self.batch_size = batch_size
        super().__init__(space)

    def _optimize(self, acquisition: Acquisition, context_manager: ContextManager) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of abstract method.
        Taking into account gradients if acquisition supports them.
        See AcquisitionOptimizerBase._optimizer for parameter descriptions.
        See class docstring for implementation details.
        """
        x_min, fx_min = None, None
        for i in range(self.batch_size):
            acquisition.reset(acquisition.model)
            x_min_i, fx_min_i = super()._optimize(acquisition, context_manager)
            
            if x_min is None:
                x_min, fx_min = x_min_i, fx_min_i
            else:
                x_min, fx_min = np.concatenate((x_min, x_min_i), axis=1), fx_min + fx_min_i
        return x_min, fx_min
    
    def optimize(self, acquisition: Acquisition, context: Optional = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimizes the acquisition function.
        :param acquisition: The acquisition function to be optimized
        :param context: Optimization context.
                        Determines whether any variable values should be fixed during the optimization
        :return: Tuple of (location of maximum, acquisition value at maximizer)
        """
        if context is None:
            context = dict()
        else:
            self._validate_context_parameters(context)
        context_manager = ContextManager(self.space, context)
        max_x, max_value = self._optimize(acquisition, context_manager)

        # Optimization might not match any encoding exactly
        # Rounding operation here finds the closest encoding
        rounded_max_x = np.concatenate(tuple(self.space.round(max_x[:,i*self.batch_size:(i+1)*self.batch_size]) for i in range(self.batch_size)), axis=1) 

        return max_x, max_value
        