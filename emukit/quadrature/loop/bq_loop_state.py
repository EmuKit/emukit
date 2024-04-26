# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional

import numpy as np

from ...core.loop.loop_state import LoopState, create_loop_state
from ...core.loop.user_function_result import UserFunctionResult


class BQLoopState(LoopState):
    """Contains the state of the BQ loop, which includes a history of all function evaluations and integral mean and
    variance estimates.

    :param initial_results: The results from previous integrand evaluations.

    """

    def __init__(self, initial_results: List[UserFunctionResult]) -> None:

        super().__init__(initial_results)

        self.integral_means = []
        self.integral_vars = []

    def update_integral_stats(self, integral_mean: float, integral_var: float) -> None:
        """Adds the latest integral mean and variance to the loop state.

        :param integral_mean: The latest integral mean estimate.
        :param integral_var: The latest integral variance.
        """
        self.integral_means.append(integral_mean)
        self.integral_vars.append(integral_var)


def create_bq_loop_state(x_init: np.ndarray, y_init: np.ndarray, **kwargs) -> BQLoopState:
    """Creates a BQ loop state object using the provided data.

    :param x_init: x values for initial function evaluations. Shape: (n_initial_points x n_input_dims)
    :param y_init: y values for initial function evaluations. Shape: (n_initial_points x n_output_dims)
    :param kwargs: extra outputs observed from a function evaluation. Shape: (n_initial_points x n_dims)
    :return: The BQ loop state.
    """

    loop_state = create_loop_state(x_init, y_init, **kwargs)
    return BQLoopState(loop_state.results)
