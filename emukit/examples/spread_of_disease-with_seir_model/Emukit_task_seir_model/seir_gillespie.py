# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from .base_models import SEIR
from .gillespie_base import GillespieBase


class SEIRGillespie(GillespieBase):
    """
    Stochastic Gillespie simulations of the SEIR model

    The model (with birth and death rates, as opposed to ours) is described at:
    https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model
    """

    def __init__(self, model: SEIR):
        """
        :param model: A SEIR model
        """
        super().__init__(model)

    def _get_initial_state(self) -> np.ndarray:
        return np.asarray([self.model.N - 1, 0, 1])

    def _get_state_index_infected(self) -> int:
        return int(2)

    def _get_possible_state_updates(self):
        """possible updates of compartment counts"""
        return np.array([[-1, 1, 0], [0, -1, 1], [0, 0, -1]], dtype=np.int)

    def _get_current_rates(self, state):
        """
        Returns an array of the current rates of infection/incubation/recovery (1/2/3),
        i.e. the un-normalized probabilities for these events to occur next
        """
        rate_1 = self.model.alpha * state[0] * state[2] / self.model.N
        rate_2 = self.model.beta * state[1]
        rate_3 = state[2]
        return np.asarray([rate_1, rate_2, rate_3])
