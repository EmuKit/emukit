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


class UserFunction(abc.ABC, Callable):
    """ The user supplied function is interrogated as part of the outer loop """
    @abc.abstractmethod
    def evaluate(self, X: np.ndarray) -> List[UserFunctionResult]:
        pass

    def __call__(self, X: np.ndarray) -> List[UserFunctionResult]:
        return self.evaluate(X)


class UserFunctionWrapper(UserFunction):
    """ Wraps a user-provided python function. """
    def __init__(self, f: Callable, extra_output_names: List[str] = None):
        """
        :param f: A python function that takes in a 2d numpy ndarray of inputs and returns a either a 2d numpy array
                  of function outputs or a tuple of (outputs, auxillary_output_1, auxilary_output_2, ...)
                  where all outputs are 2d
        :param extra_output_names: If the function f returns a tuple, the first output should be the value of the
                                   objective, which will be named "Y", names for subsequent outputs should be included
                                   in this list.
        """
        self.f = f
        self.extra_output_names = [] if extra_output_names is None else extra_output_names

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
            extra_outputs = outputs[1:]
        elif isinstance(outputs, np.ndarray):
            user_fcn_outputs = outputs
            extra_outputs = tuple()
        else:
            raise ValueError("User provided function should return a tuple or an ndarray, "
                             "{} received".format(type(outputs)))

        # Validate number of outputs returned by the user function
        if len(extra_outputs) != len(self.extra_output_names):
            raise ValueError('User function provided {} outputs but UserFunctionWrapper expected {}'.format(
                len(extra_outputs) + 1, len(self.extra_output_names) + 1))

        if user_fcn_outputs.ndim != 2:
            raise ValueError("User function should return 2d array or a tuple of 2d arrays as an output, "
                             "actual output dimensionality is {}".format(outputs.ndim))

        results = []
        for i in range(user_fcn_outputs.shape[0]):
            kwargs = dict([(name, val[i]) for name, val in zip(self.extra_output_names, extra_outputs)])
            results.append(UserFunctionResult(inputs[i], user_fcn_outputs[i], **kwargs))

        return results


class MultiSourceFunctionWrapper(UserFunction):
    """
    Wraps a list of python functions that each correspond to different information source.
    """

    def __init__(self, f: List, source_index: int=-1, extra_output_names: List[str] = None) -> None:
        """
        :param f: A list of python function that take in a 2d numpy ndarrays of inputs and return 2d numpy ndarrays
                  of outputs.
        :param source_index: An integer indicating which column of X contains the index of the information source.
                             Default to the last dimension of the input.
        :param extra_output_names: If the function f returns a tuple, the first output should be the value of the
                                   objective, which will be named "Y", names for subsequent outputs should be included
                                   in this list.
        """
        self.f = f
        self.source_index = source_index
        self.extra_output_names = [] if extra_output_names is None else extra_output_names

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
        indices, outputs, extra_outputs = [], [], []
        source_indices = inputs[:, self.source_index]
        source_inputs = np.delete(inputs, self.source_index, axis=1)
        for i_source in range(n_sources):
            # Find inputs at that source
            this_source_input_indices = np.flatnonzero(source_indices == i_source)
            indices.append(this_source_input_indices)
            this_source_inputs = source_inputs[this_source_input_indices]
            this_outputs = self.f[i_source](this_source_inputs)

            if isinstance(this_outputs, tuple):
                outputs.append(this_outputs[0])
                extra_outputs.append(this_outputs[1:])

                # Check correct number of outputs from user function
                if len(extra_outputs[-1]) != len(self.extra_output_names):
                    raise ValueError('Expected {} outputs from user function but got {}'.format(
                        len(self.extra_output_names) + 1, len(extra_outputs[-1]) + 1))
            elif isinstance(this_outputs, np.ndarray):
                outputs.append(this_outputs)

                # Check correct number of outputs from user function
                if len(self.extra_output_names) != 0:
                    raise ValueError('Expected {} output from user function but got 1'.format(
                        len(self.extra_output_names) + 1))

                # Dummy extra outputs - won't be used below
                extra_outputs.append(tuple())
            else:
                raise ValueError("User provided function should return a tuple or an ndarray, "
                                 "{} received".format(type(this_outputs)))

        sort_indices = np.argsort(np.concatenate(indices, axis=0))
        outputs = np.concatenate(outputs, axis=0)

        # Concatenate list of lists to single list
        n_extra_outputs = len(self.extra_output_names)
        extra_output_lists = [[] for _ in range(n_extra_outputs)]
        for i_source in range(n_sources):
            for i_output in range(n_extra_outputs):
                extra_output_lists[i_output].extend(extra_outputs[i_source][i_output])

        results = []
        for i, idx_sorted in enumerate(sort_indices):
            # Put extra outputs into a dictionary so we can pass them as key word arguments
            kwargs = dict([(name, val[idx_sorted]) for name, val in zip(self.extra_output_names, extra_output_lists)])
            results.append(UserFunctionResult(inputs[i], outputs[idx_sorted], **kwargs))
        return results
