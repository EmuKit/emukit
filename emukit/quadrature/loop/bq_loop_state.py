# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional

import numpy as np

from ...core.loop.loop_state import LoopState, create_loop_state
from ...core.loop.user_function_result import UserFunctionResult


class BQLoopState(LoopState):
    """Contains the state of the BQ loop, which includes a history of all function evaluations and integral mean and
    variance estimates.

    :param initial_results:
    :param initial_integral_means:
    :param initial_integral_vars:

    :raises ValueError: If ``initial_integral_means`` and ``initial_integral_vars`` have different length.

    """
    def __init__(
        self,
        initial_results: List[UserFunctionResult],
        initial_integral_means: Optional[List[float]] = None,
        initial_integral_vars: Optional[List[float]] = None,
    ) -> None:
        super().__init__(initial_results)

        self.integral_means = initial_integral_means
        if self.integral_means is None:
            self.integral_means = []

        self.integral_vars = initial_integral_vars
        if self.integral_vars is None:
            self.integral_vars = []

        if len(self.integral_means) != len(self.integral_vars):
            raise ValueError(
                f"initial integral means list must have same length ({len(self.integral_means)}) "
                f"as initial integral variances list ({len(self.integral_vars)})."
            )

    def update_integral_stats(self, integral_mean: float, integral_var: float) -> None:
        """Adds the latest integral mean and variance to the loop state.

        :param integral_mean: The latest integral mean estimate.
        :param integral_var: The latest integral variance.
        """
        self.integral_means.append(integral_mean)
        self.integral_vars.append(integral_var)


def create_bq_loop_state(
    x_init: np.ndarray,
    y_init: np.ndarray,
    initial_integral_means: Optional[List[float]] = None,
    initial_integral_vars: Optional[List[float]] = None,
    **kwargs,
) -> BQLoopState:
    """Creates a bq loop state object using the provided data.

    :param x_init: x values for initial function evaluations. Shape: (n_initial_points x n_input_dims)
    :param y_init: y values for initial function evaluations. Shape: (n_initial_points x n_output_dims)
    :param initial_integral_means:
    :param initial_integral_vars:
    :param kwargs: extra outputs observed from a function evaluation. Shape: (n_initial_points x n_dims)
    :return: The bq loop state.
    """

    loop_state = create_loop_state(x_init, y_init, **kwargs)
    return BQLoopState(loop_state.results, initial_integral_means, initial_integral_vars)
