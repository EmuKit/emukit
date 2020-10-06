# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging

import numpy as np

from .. import ParameterSpace
from ..acquisition import Acquisition
from .context_manager import ContextManager

_log = logging.getLogger(__name__)


class AnchorPointsGenerator(object):
    """
    Anchor points are the points from which the optimization of the acquisition function is initialized.

    This base class is for generating such points, and the sub-classes will implement different logic of how the
    points should be selected
    """
    def __init__(self, space: ParameterSpace, num_samples: int):
        """

        :param space: Parameter space describing the input domain of the model
        :param num_samples: Number of initial samples to draw uniformly from the input domain. These points are then
                            evaluated according to logic implemented in the subclasses, and the best are picked
        """
        self.space = space
        self.num_samples = num_samples

    def get_anchor_point_scores(self, X: np.ndarray) -> np.ndarray:
        """
        This abstract method should contain the logic to ascribe scores to different points in the input domain.
        Points with higher scores will be chosen over points with lower scores.

        :param X: (n_samples x n_inputs_dims) arrays containing the points at which to evaluate the anchor point scores
        :return: Array containing score for each input point
        """
        raise NotImplementedError('get_anchor_point_scores is not implemented in the parent class.')

    def get(self, num_anchor: int = 5, context_manager: ContextManager = None) -> np.ndarray:
        """
        :param num_anchor: Number of points to return
        :param context_manager: Describes any fixed parameters in the optimization
        :return: A (num_anchor x n_dims) array containing the anchor points
        """
        if num_anchor > self.num_samples:
            raise ValueError(f"Cannot return more points than were originally sampled, got {num_anchor} num_anchor for {self.num_samples} points sampled.")

        # We use the context handler to remove duplicates only over the non-context variables
        if context_manager is not None:
            space = context_manager.contextfree_space
        else:
            space = self.space

        # Generate initial design
        X = space.sample_uniform(self.num_samples)

        # Add context variables
        if context_manager:
            X = context_manager.expand_vector(X)
        scores = self.get_anchor_point_scores(X)
        sorted_idxs = np.argsort(scores)[::-1]
        anchor_points = X[sorted_idxs[:min(len(scores), num_anchor)], :]

        return anchor_points


class ObjectiveAnchorPointsGenerator(AnchorPointsGenerator):
    """
    This anchor points generator chooses points where the acquisition function is highest
    """
    def __init__(self, space: ParameterSpace, acquisition: Acquisition, num_samples: int=1000):
        """
        :param space: The parameter space describing the input domain of the non-context variables
        :param acquisition: The acquisition function
        :param num_samples: The number of points at which the anchor point scores are calculated
        """
        self.acquisition = acquisition
        super(ObjectiveAnchorPointsGenerator, self).__init__(space, num_samples)

    def get_anchor_point_scores(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: The samples at which to evaluate the criterion
        :return:
        """
        are_constraints_satisfied = np.all([np.ones(X.shape[0])] + [c.evaluate(X) for c in self.space.constraints], axis=0)
        scores = np.zeros((X.shape[0],))
        scores[~are_constraints_satisfied] = -np.inf
        scores[are_constraints_satisfied] = self.acquisition.evaluate(X[are_constraints_satisfied, :])[:, 0]
        return scores
