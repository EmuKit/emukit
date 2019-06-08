# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List

import numpy as np

from .user_function_result import UserFunctionResult


class UnknownConstraintLoopState(object):
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

    @property
    def Y_constraint(self) -> np.ndarray:
        """
        :return: Outputs for all constraint evaluations in a 2d array: number of points by output dimensions.
        """
        return np.array([result.Y_constraint for result in self.results])


def create_constrained_loop_state(x_init: np.ndarray, y_init: np.ndarray, y_constraint: np.ndarray) -> UnknownConstraintLoopState:
    """
    Creates a loop state object using the provided data

    :param x_init: x values for initial function evaluations.
    :param y_init: y values for initial function evaluations
    :param y_init: y values for initial constraint evaluations
    """
    if x_init.shape[0] != y_init.shape[0]:
        error_message = "X and Y should have the same length. Actual length x_init {}, y_init {}".format(
            x_init.shape[0], y_init.shape[0])
        raise ValueError(error_message)

    initial_results = []
    for x, y, y_constraint in zip(x_init, y_init, y_constraint):
            initial_results.append(UserFunctionResult(x, y, y_constraint))

    return UnknownConstraintLoopState(initial_results)
