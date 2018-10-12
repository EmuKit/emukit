# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import abc

from . import LoopState


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
        :param i_max: Maximum number of function observations within the loop, excluding initial points
        """
        self.i_max = i_max

    def should_stop(self, loop_state: LoopState) -> bool:
        """
        :param loop_state: Object that contains current state of the loop
        :return: True if maximum number of iterations has been reached
        """
        return loop_state.iteration >= self.i_max
