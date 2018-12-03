# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from .acquisitions import ModelVariance
from ...core.acquisition import Acquisition
from ...core.interfaces.models import IModel
from ...core.loop import OuterLoop, SequentialPointCalculator, FixedIntervalUpdater
from ...core.loop.candidate_point_calculators import GreedyBatchPointCalculator
from ...core.loop.loop_state import create_loop_state
from ...core.optimization import AcquisitionOptimizer
from ...core.parameter_space import ParameterSpace


class ExperimentalDesignLoop(OuterLoop):
    def __init__(self, space: ParameterSpace, model: IModel, acquisition: Acquisition = None, update_interval: int = 1,
                 batch_size: int=1):
        """
        An outer loop class for use with Experimental design

        :param space: Definition of domain bounds to collect points within
        :param model: The model that approximates the underlying function
        :param acquisition: experimental design acquisition function object. Default: ModelVariance acquisition
        :param update_interval: How many iterations pass before next model optimization
        :param batch_size: Number of points to collect in a batch. Defaults to one.
        """

        if acquisition is None:
            acquisition = ModelVariance(model)

        # This AcquisitionOptimizer object deals with optimizing the acquisition to find the next point to collect
        acquisition_optimizer = AcquisitionOptimizer(space)

        # Construct emukit classes
        if batch_size == 1:
            candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)
        elif batch_size > 1:
            candidate_point_calculator = \
                GreedyBatchPointCalculator(model, acquisition, acquisition_optimizer, batch_size)
        else:
            raise ValueError('Batch size value of ' + str(batch_size) + ' is invalid.')

        model_updater = FixedIntervalUpdater(model, update_interval)
        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updater, loop_state)

        self.model = model
