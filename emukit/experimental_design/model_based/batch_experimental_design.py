# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel
from emukit.core.loop import CandidatePointCalculator, LoopState
from emukit.core.optimization import AcquisitionOptimizer


class GreedyBatchPointCalculator(CandidatePointCalculator):
    """
    Batch point calculator for use in batch experimental design. This point calculator calculates the first point in
    the batch then adds this as a fake observation in the model with a Y value equal to the mean prediction. The model
    is reset with the original data at the end of collecting a batch but if you use a model where training the model
    with the same data leads to different predictions, the model behaviour will be modified.
    """
    def __init__(self, model: IModel, acquisition: Acquisition, acquisition_optimizer: AcquisitionOptimizer,
                 batch_size: int=1):
        """
        :param model: Model that is used by the acquisition function
        :param acquisition: Acquisition to be optimized to find each point in batch
        :param acquisition_optimizer: Acquisition optimizer that optimizes acquisition function to find each point in batch
        :param batch_size: Number of points to calculate in batch
        """
        if not isinstance(batch_size, int):
            raise ValueError('Batch size should be an integer')
        self.model = model
        self.acquisition = acquisition
        self.acquisition_optimizer = acquisition_optimizer
        self.batch_size = batch_size

    def compute_next_points(self, loop_state: LoopState) -> np.ndarray:
        """
        :param loop_state: Object containing history of the loop
        :return: 2d array of size (batch_size x input dimensions) of new points to evaluate
        """
        new_xs = []
        original_data = (self.model.X, self.model.Y)
        for _ in range(self.batch_size):
            new_x, _ = self.acquisition_optimizer.optimize(self.acquisition)
            new_xs.append(new_x)
            new_y = self.model.predict(new_x)[0]

            # Add new point as fake observation in model
            all_x = np.concatenate([self.model.X, new_x], axis=0)
            all_y = np.concatenate([self.model.Y, new_y], axis=0)
            self.model.update_data(all_x, all_y)

        # Reset data
        self.model.update_data(*original_data)
        return np.concatenate(new_xs, axis=0)
