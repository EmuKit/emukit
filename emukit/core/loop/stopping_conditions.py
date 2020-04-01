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

    def __and__(self, other: 'StoppingCondition') -> 'And':
        """
        Overloads self & other
        """
        return And(self, other)

    def __or__(self, other: 'StoppingCondition') -> 'Or':
        """
        Overloads self | other
        """
        return Or(self, other)

    @abc.abstractmethod
    def should_stop(self, loop_state: LoopState) -> bool:
        """
        :param loop_state: Object that contains current state of the loop
        :return: Whether to stop collecting new data
        """
        pass


class And(StoppingCondition):
    """
    Logical AND of two stopping conditions
    """
    def __init__(self, left: StoppingCondition, right: StoppingCondition):
        """
        :param left: One stopping condition in AND
        :param right: Another stopping condition in AND
        """
        self.left = left
        self.right = right

    def should_stop(self, loop_state: LoopState) -> bool:
        """
        Evaluate logical AND of two stopping conditions

        :param loop_state: Object that contains current state of the loop
        :return: Whether to stop collecting new data
        """
        return self.left.should_stop(loop_state) and self.right.should_stop(loop_state)


class Or(StoppingCondition):
    """
    Logical OR of two stopping conditions
    """
    def __init__(self, left: StoppingCondition, right: StoppingCondition):
        """
        :param left: One stopping condition in OR
        :param right: Another stopping condition in OR
        """
        self.left = left
        self.right = right

    def should_stop(self, loop_state: LoopState) -> bool:
        """
        Evaluate logical OR of two stopping conditions

        :param loop_state: Object that contains current state of the loop
        :return: Whether to stop collecting new data
        """
        return self.left.should_stop(loop_state) or self.right.should_stop(loop_state)


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
        should_stop = loop_state.iteration >= self.i_max
        if should_stop is True:
            _log.info("Stopped after {} evaluations".format(self.i_max))
        return should_stop


class ConvergenceStoppingCondition(StoppingCondition):
    """ Stops once we choose a point within eps of a previous
            point (with respect to euclidean norm). Close evaluations
            can suggest convergence of the optimization for problems 
            with low observation noise.
            """
    def __init__(self, eps: float) -> None:
        """
        :param eps: minimum distance between
        two consecutive x's to keep running the model
        """
        self.eps = eps

    def should_stop(self, loop_state: LoopState) -> bool:
        """
        :param loop_state: Object that contains current state of the loop
        :return: True if the euclidean distance between the last two evaluations
                    is smaller than the specified eps.
        """
        if loop_state.iteration < 2:
            # less than 2 evaluations so cannot calculate distance
            return False
        should_stop = np.linalg.norm(loop_state.X[-1, :]-loop_state.X[-2, :]).item() <= self.eps
        if should_stop is True:
            _log.info("Stopped as consecutive evaluations are within {}".format(self.eps))
        return should_stop
