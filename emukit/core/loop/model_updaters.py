# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import abc
from typing import Callable

from . import LoopState
from ..interfaces import IModel


import logging
_log = logging.getLogger(__name__)


class ModelUpdater(abc.ABC):
    @abc.abstractmethod
    def update(self, loop_state: LoopState) -> None:
        """
        Updates the training data of the model. Chooses whether to update hyper-parameters and how to do it

        :param loop_state: Object that contains current state of the loop
        """
        pass


class FixedIntervalUpdater(ModelUpdater):
    """ Updates hyper-parameters every nth iteration, where n is defined by the user """
    def __init__(self, model: IModel, interval: int=1, targets_extractor_fcn: Callable=None) -> None:
        """
        :param model: Emukit emulator model
        :param interval: Number of function evaluations between optimizing model hyper-parameters
        :param targets_extractor_fcn: A function that takes in loop state and returns the training targets.
                                      Defaults to a function returning loop_state.Y
        """
        self.model = model
        self.interval = interval

        if targets_extractor_fcn is None:
            self.targets_extractor_fcn = lambda loop_state: loop_state.Y
        else:
            self.targets_extractor_fcn = targets_extractor_fcn

    def update(self, loop_state: LoopState) -> None:
        """
        :param loop_state: Object that contains current state of the loop
        """
        targets = self.targets_extractor_fcn(loop_state)
        self.model.set_data(loop_state.X, targets)
        if (loop_state.iteration % self.interval) == 0:
            _log.info("Updating parameters of the model")
            self.model.optimize()
