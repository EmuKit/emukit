# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ...core.loop.outer_loop import OuterLoop
from ...core.loop.candidate_point_calculators import RandomSampling
from ...core.loop.loop_state import create_loop_state
from ...core.parameter_space import ParameterSpace
from ...core.loop.model_updaters import NoopModelUpdater


class RandomSearch(OuterLoop):
    def __init__(self, space: ParameterSpace, x_init: np.ndarray = None,
                 y_init: np.ndarray = None, cost_init: np.ndarray = None):

        """
        Simple loop to perform random search where in each iteration points are sampled uniformly at random
        over the input space.

        :param space: Input space where the optimization is carried out.
        :param x_init: 2d numpy array of shape (no. points x no. input features) of initial X data
        :param y_init: 2d numpy array of shape (no. points x no. targets) of initial Y data
        :param cost_init: 2d numpy array of shape (no. points x no. targets) of initial cost of each function evaluation
        """

        model_updaters = NoopModelUpdater()

        candidate_point_calculator = RandomSampling(parameter_space=space)

        if x_init is not None and y_init is not None:
            loop_state = create_loop_state(x_init, y_init, cost=cost_init)
        else:
            loop_state = None

        super().__init__(candidate_point_calculator, model_updaters, loop_state=loop_state)
