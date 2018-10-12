# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import abc

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
    def __init__(self, model: IModel, interval: int = 1) -> None:
        """
        :param model: Emukit emulator model
        :param interval: Number of function evaluations between optimizing model hyper-parameters
        """
        self.model = model
        self.interval = interval

    def update(self, loop_state: LoopState) -> None:
        """
        :param loop_state: Object that contains current state of the loop
        """
        self.model.update_data(loop_state.X, loop_state.Y)
        if (loop_state.iteration % self.interval) == 0:
            _log.info("Updating parameters of the model")
            self.model.optimize()
