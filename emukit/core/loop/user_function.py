# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
This file contains the "UserFunction" base class and implementations

The user function is the objective function in optimization, the integrand in quadrature or the function to be learnt
in experimental design.
"""

import abc
from typing import Callable, List

import numpy as np

from .user_function_result import UserFunctionResult


import logging
_log = logging.getLogger(__name__)


class UserFunction(abc.ABC):
    """ The user supplied function is interrogated as part of the outer loop """
    @abc.abstractmethod
    def evaluate(self, X: np.ndarray) -> List[UserFunctionResult]:
        pass


class UserFunctionWrapper(UserFunction):
    """ Wraps a user-provided python function. """
    def __init__(self, f: Callable):
        """
        :param f: A python function that takes in a 2d numpy ndarray of inputs and returns a either a 2d numpy array
                  of function outputs or a tuple of (outputs, evaluation_costs) where both the outputs and the
                  cost are 2d
        """
        self.f = f

    def evaluate(self, inputs: np.ndarray) -> List[UserFunctionResult]:
        """
        Evaluates python function by providing it with numpy types and converts the output
        to a List of UserFunctionResults

        :param inputs: List of function inputs at which to evaluate function
        :return: List of function results
        """
        if inputs.ndim != 2:
            raise ValueError("User function should receive 2d array as an input, "
                             "actual input dimensionality is {}".format(inputs.ndim))

        _log.info("Evaluating user function for {} point(s)".format(inputs.shape[0]))
        outputs = self.f(inputs)

        if isinstance(outputs, tuple):
            user_fcn_outputs = outputs[0]
            cost = outputs[1]
        elif isinstance(outputs, np.ndarray):
            user_fcn_outputs = outputs
            cost = [None] * user_fcn_outputs.shape[0]
        else:
            raise ValueError("User provided function should return a tuple or an ndarray, "
                             "{} received".format(type(outputs)))

        if user_fcn_outputs.ndim != 2:
            raise ValueError("User function should return 2d array or a tuple of 2d arrays as an output, "
                             "actual output dimensionality is {}".format(outputs.ndim))

        results = []
        for x, y, c in zip(inputs, user_fcn_outputs, cost):
            results.append(UserFunctionResult(x, y.flatten(), c))
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

        :param inputs: A list of inputs to evaluate the function at
                       with information source index appended as last column
        :return: A list of function outputs
        """

        if inputs.ndim != 2:
            raise ValueError("User function should receive 2d array as an input, "
                             "actual input dimensionality is {}".format(inputs.ndim))

        n_sources = len(self.f)

        _log.info("Evaluating multi-source user function for {} point(s)".format(inputs.shape[0]))
        # Run each source function for all inputs at that source
        outputs = []
        for i_source in range(n_sources):
            # Find inputs at that source
            is_this_source = inputs[:, self.source_index] == i_source
            this_source_inputs = np.delete(inputs[is_this_source, :], self.source_index, axis=1)
            outputs.append(self.f[i_source](this_source_inputs))

        outputs_array = np.concatenate(outputs, axis=0)
        results = []
        for x, y in zip(inputs, outputs_array):
            results.append(UserFunctionResult(x, y))
        return results
