# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List

import numpy as np

from .user_function_result import UserFunctionResult


class LoopState(object):
    """
    Contains the state of the loop, which includes a history of all function evaluations
    """

    def __init__(self, initial_results: List[UserFunctionResult]) -> None:
        """
        :param initial_results: The function results from previous function evaluations
        """
        self.results = initial_results
        self.iteration = 0

    def update(self, results: List[UserFunctionResult]) -> None:
        """
        :param results: The latest function results since last update
        """
        if not results:
            raise ValueError("Cannot update state with empty result list.")

        self.iteration += 1
        self.results += results

    @property
    def X(self) -> np.ndarray:
        """
        :return: Function inputs for all function evaluations in a 2d array: number of points by input dimensions.
        """
        return np.array([result.X for result in self.results])

    @property
    def Y(self) -> np.ndarray:
        """
        :return: Function outputs for all function evaluations in a 2d array: number of points by output dimensions.
        """
        return np.array([result.Y for result in self.results])

    def __getattr__(self, item) -> np.array:
        """
        Overriding this method allows us to customise behaviour for accessing attributes. We use this to allow arbitrary
        fields in the loop state. These are usually extra outputs from the user function such as cost, constraint values
        etc. These fields are stored in each individual "UserFunctionResult" object in the "extra_outputs" dictionary

        :param item: The name of the item to acquire. Must match the key value in the "extra_outputs" dictionary in the
                     stored "UserFunctionResults" objects
        :return: The specified output for all function evaluations in a 2d array of size (n_points x n_dimensions)
        """
        # check key appears in all results objects
        is_valid = all([item in res.extra_outputs for res in self.results])
        if not is_valid:
            raise ValueError('{} not found in results object'.format(item))
        return np.array([result.extra_outputs[item] for result in self.results])


def create_loop_state(x_init: np.ndarray, y_init: np.ndarray, **kwargs) -> LoopState:
    """
    Creates a loop state object using the provided data

    :param x_init: x values for initial function evaluations. Shape: (n_initial_points x n_input_dims)
    :param y_init: y values for initial function evaluations. Shape: (n_initial_points x n_output_dims)
    :param kwargs: extra outputs observed from a function evaluation. Shape: (n_initial_points x n_dims)
    """
    if x_init.shape[0] != y_init.shape[0]:
        error_message = "X and Y should have the same length. Actual length x_init {}, y_init {}".format(
            x_init.shape[0], y_init.shape[0])
        raise ValueError(error_message)

    for key, value in kwargs.items():
        if value.shape[0] != x_init.shape[0]:
            raise ValueError('Expected keyword argument {} to have length {} but actual length is {}'.format(
                key, x_init.shape[0], value.shape[0]))

    initial_results = []
    for i in range(x_init.shape[0]):
        kwargs_dict = dict([(key, vals[i]) for key, vals in kwargs.items()])
        initial_results.append(UserFunctionResult(x_init[i], y_init[i], **kwargs_dict))

    return LoopState(initial_results)
