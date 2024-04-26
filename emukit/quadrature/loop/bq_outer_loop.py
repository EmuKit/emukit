# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Union
from ...core.loop import OuterLoop

from ...core.loop.candidate_point_calculators import CandidatePointCalculator
from ...core.loop.model_updaters import ModelUpdater

from .bq_loop_state import BQLoopState


class BQOuterLoop(OuterLoop):
    """Base class for a Bayesian quadrature outer loop.

    :param candidate_point_calculator: Finds next point(s) to evaluate.
    :param model_updater: Updates the model with the new data and fits the model hyper-parameters.
    :param loop_state: Object that keeps track of the history of the BQ loop. Default is None, resulting in empty
                       initial state.
    """

    def __init__(
        self,
        candidate_point_calculator: CandidatePointCalculator,
        model_updater: Union[ModelUpdater, List[ModelUpdater]],
        loop_state: BQLoopState = None,
    ):
        super().__init__(candidate_point_calculator, model_updater, loop_state)

    def _update_loop_state_custom(self) -> None:
        integral_mean, integral_var = self.model.integrate()
        self.loop_state.update_integral_stats(integral_mean, integral_var)
