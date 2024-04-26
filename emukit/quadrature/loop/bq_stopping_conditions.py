# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

import numpy as np

from ...core.loop.stopping_conditions import StoppingCondition

from .bq_loop_state import BQLoopState
_log = logging.getLogger(__name__)


class CoefficientOfVariationStoppingCondition(StoppingCondition):
    r"""Stops once the coefficient of variation (COV) falls below a threshold.

    The COV is given by

    .. math::
        COV = \frac{\sigma}{\mu}

    with :math:`\mu` and :math:`\sigma^2` the current mean and standard deviation of integral according to the
    BQ posterior model.

    :param eps: Threshold under which the COV must fall.
    :param delay: Number of times the stopping condition needs to be true in a row in order to stop. Defaults to 1.

    """

    def __init__(self, eps: float, delay: int = 1) -> None:

        if delay < 1:
            raise ValueError(f"delay ({delay}) must be and integer greater than zero.")

        self.eps = eps
        self.delay = delay
        self.times_true = 0  # counts how many times stopping had been triggered in a row

    def should_stop(self, loop_state: BQLoopState) -> bool:
        if len(loop_state.integral_means) < 1:
            return False

        m = loop_state.integral_means[-1]
        v = loop_state.integral_vars[-1]
        should_stop = (np.sqrt(v) / m) < self.eps

        if should_stop:
            self.times_true += 1
        else:
            self.times_true = 0

        print(np.sqrt(v) / m)
        print(should_stop, self.times_true)
        should_stop = should_stop and (self.times_true >= self.delay)
        print(should_stop)
        print()

        if should_stop:
            _log.info(f"Stopped as coefficient of variation is below threshold of {self.eps}.")
        return should_stop
