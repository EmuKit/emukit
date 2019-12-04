# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import abc
import numpy as np
import math

from . import LoopState

import logging
_log = logging.getLogger(__name__)


class StoppingCondition(abc.ABC):
    """ Chooses whether to stop the optimization based on the loop state """
    @abc.abstractmethod
    def should_stop(self, loop_state: LoopState) -> bool:
        """
        :param loop_state: Object that contains current state of the loop
        :return: Whether to stop collecting new data
        """
        pass


class FixedIterationsStoppingCondition(StoppingCondition):
    """ Stops after a fixed number of iterations """
    def __init__(self, i_max: int) -> None:
        """
        :param i_max: Maximum number of function
        observations within the loop, excluding initial points
        """
        self.i_max = i_max

    def should_stop(self, loop_state: LoopState) -> bool:
        """
        :param loop_state: Object that contains current state of the loop
        :return: True if maximum number of iterations has been reached
        """
        status = loop_state.iteration >= self.i_max
        if status is True:
            _log.info("Stopped after {} Evaluations".format(self.i_max))
        return status


class ConvergenceStoppingCondition(StoppingCondition):
    """ Stops once we choose a point within eps of a previous
    point (with respect to euclidean norm), a notion of convergence"""
    def __init__(self, eps: float) -> None:
        """
        :param eps: minimum distance between
        two consecutive x's to keep running the model
        """
        self.eps = eps

    def should_stop(self, loop_state: LoopState) -> bool:
        """
        :param loop_state: Object that contains current state of the loop
        :return: True if consecutive x's are too close
        """
        if loop_state.iteration < 2:
            # less than 2 evaluations so cannot calculate distance
            return False
        status = math.sqrt(sum((loop_state.X[-1, :] - loop_state.X[-2, :]) ** 2)) <= self.eps
        if status is True:
            _log.info("Stopped as consecutive evaluations are within {}".format(self.eps))
        return status
