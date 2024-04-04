# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from .base_models import SIR
from .gillespie_base import GillespieBase


class SIRGillespie(GillespieBase):
    """
    Stochastic Gillespie simulations of the SIR model

    Some of the code has been adapted from a tutorial at
    http://be150.caltech.edu/2016/handouts/gillespie_simulation.html
    (retrieved on Aug. 23, 2018)
    """

    def __init__(self, model: SIR):
        """x
        :param model: A SIR model
        """
        super().__init__(model)

    def _get_initial_state(self) -> np.ndarray:
        return np.asarray([self.model.N - 1, 1])

    def _get_state_index_infected(self):
        return int(1)

    def _get_possible_state_updates(self) -> np.ndarray:
        """possible updates of compartment counts"""
        return np.array([[-1, 1], [0, -1]], dtype=np.int)

    def _get_current_rates(self, state: np.ndarray) -> np.ndarray:
        """
        Returns an array of the current rates of infection/recovery (1/2),
        i.e. the un-normalized probabilities for these events to occur next
        """
        rate_1 = self.model.alpha * state[0] * state[1] / self.model.N
        rate_2 = state[1]
        return np.asarray([rate_1, rate_2])
