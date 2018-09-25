"""
This file contains the "OuterFunction" base class and implementations

The outer function is the objective function in optimization, the integrand in quadrature or the function to be learnt
in experimental design.
"""

import abc
from typing import Callable, List

import numpy as np

from .user_function_result import UserFunctionResult


class UserFunction(abc.ABC):
    """ The user supplied function is interrogated as part of the outer loop """
    @abc.abstractmethod
    def evaluate(self, X: np.ndarray) -> List[UserFunctionResult]:
        pass


class UserFunctionWrapper(UserFunction):
    """ Wraps a user-provided python function. """
    def __init__(self, f: Callable):
        """
        :param f: A python function that takes in a 2d numpy ndarray of inputs and returns a 2d numpy ndarray of outputs.
        """
        self.f = f

    def evaluate(self, inputs: np.ndarray) -> List[UserFunctionResult]:
        """
        Evaluates python function by providing it with numpy types and converts the output to a List of OuterFunctionResults

        :param inputs: List of function inputs at which to evaluate function
        :return: List of function results
        """
        if inputs.ndim != 2:
            raise ValueError("User function should receive 2d array as an input, actual input dimensionality is {}".format(inputs.ndim))

        outputs = self.f(inputs)

        if outputs.ndim != 2:
            raise ValueError("User function should return 2d array as an output, actual output dimensionality is {}".format(outputs.ndim))

        results = []
        for x, y in zip(inputs, outputs):
            results.append(UserFunctionResult(x, y))
        return results

class MultiSourceFunctionWrapper(UserFunction):
    """
    Wraps a list of python functions that each correspond to different information source.
    """

    def __init__(self, f: List, source_index: int=-1) -> None:
        """
        :param f: A list of python function that take in a 2d numpy ndarrays of inputs and return 2d numpy ndarrays
                  of outputs.
        :param source_index: An integer indicating which column of X contains the index of the information source.
                             Default to the last dimension of the input.
        """
        self.f = f
        self.source_index = source_index

    def evaluate(self, inputs: np.ndarray) -> List[UserFunctionResult]:
        """
        Evaluates the python functions corresponding to the appropriate information source

        :param inputs: A list of inputs to evaluate the function at with information source index appended as last column
        :return: A list of function outputs
        """

        if inputs.ndim != 2:
            raise ValueError("User function should receive 2d array as an input, actual input dimensionality is {}".format(inputs.ndim))

        n_sources = len(self.f)

        # Run function for inputs at the first information source
        is_first_source = inputs[:, self.source_index] == 0
        first_source_inputs = np.delete(inputs[is_first_source, :], self.source_index, axis=1)
        first_source_outputs = self.f[0](first_source_inputs)
        outputs = np.zeros((inputs.shape[0], first_source_outputs.shape[1]))
        outputs[is_first_source, :] = first_source_outputs

        # Run each source function for all inputs at that source
        for i_source in range(1, n_sources):
            # Find inputs at that source
            is_this_source = inputs[:, self.source_index] == i_source
            this_source_inputs = np.delete(inputs[is_this_source, :], self.source_index, axis=1)
            outputs[is_this_source, :] = self.f[i_source](this_source_inputs)

        results = []
        for x, y in zip(inputs, outputs):
            results.append(UserFunctionResult(x, y))
        return results
