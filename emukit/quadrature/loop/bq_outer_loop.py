# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List, Union

from ...core.loop import OuterLoop
from ...core.loop.candidate_point_calculators import CandidatePointCalculator
from ...core.loop.model_updaters import ModelUpdater
from .bq_loop_state import QuadratureLoopState


class QuadratureOuterLoop(OuterLoop):
    """Base class for a Bayesian quadrature loop.

    :param candidate_point_calculator: Finds next point(s) to evaluate.
    :param model_updaters: Updates the model with the new data and fits the model hyper-parameters.
    :param loop_state: Object that keeps track of the history of the BQ loop. Default is None, resulting in empty
                       initial state.

    :raises ValueError: If more than one model updater is provided.

    """

    def __init__(
        self,
        candidate_point_calculator: CandidatePointCalculator,
        model_updaters: Union[ModelUpdater, List[ModelUpdater]],
        loop_state: QuadratureLoopState = None,
    ):
        if isinstance(model_updaters, list):
            raise ValueError("The BQ loop only supports a single model.")

        super().__init__(candidate_point_calculator, model_updaters, loop_state)

    def _update_loop_state(self) -> None:
        model = self.model_updaters[0].model  # only works if there is one model, but for BQ nothing else makes sense
        integral_mean, integral_var = model.integrate()
        self.loop_state.update_integral_stats(integral_mean, integral_var)
