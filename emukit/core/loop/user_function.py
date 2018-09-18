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
