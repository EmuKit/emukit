# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

import numpy as np

from .. import ParameterSpace

_log = logging.getLogger(__name__)


class InitialDesignBase(object):
    """
    Base class for all initial designs
    """

    def __init__(self, parameter_space: ParameterSpace, max_retries: int = 100):
        """
        :param parameter_space: The parameter space to generate design for.
        :param max_retries: Maximum number of retry attempts to generate valid samples when constraints are present.
                           Default is 100.
        """
        self.parameter_space = parameter_space
        self.max_retries = max_retries

    def _generate_samples(self, point_count: int) -> np.ndarray:
        """
        Generate samples without constraint checking. Should be overridden by subclasses.

        :param point_count: Number of points required.
        :return: A numpy array of generated samples, shape (point_count x space_dim)
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _check_constraints(self, samples: np.ndarray) -> np.ndarray:
        """
        Check which samples satisfy all constraints.

        :param samples: Array of shape (n_points x n_dims)
        :return: Boolean array of shape (n_points,) where True indicates the point satisfies all constraints
        """
        if not self.parameter_space.constraints:
            return np.ones(samples.shape[0], dtype=bool)

        # Start with all points being valid
        valid = np.ones(samples.shape[0], dtype=bool)

        # Check each constraint and keep only points that satisfy all
        for constraint in self.parameter_space.constraints:
            constraint_satisfaction = constraint.evaluate(samples)
            valid = valid & constraint_satisfaction

        return valid

    def get_samples(self, point_count: int) -> np.ndarray:
        """
        Generates requested amount of points that satisfy all constraints.
        Uses rejection sampling: if any constraints are present and violated,
        the entire batch is regenerated.

        :param point_count: Number of points required.
        :return: A numpy array of generated samples, shape (point_count x space_dim)
        :raises RuntimeError: If unable to generate the required number of valid points after max_retries attempts.
        """
        # If there are no constraints, just generate and return
        if not self.parameter_space.constraints:
            return self._generate_samples(point_count)

        # With constraints: use rejection sampling
        for attempt in range(self.max_retries):
            candidates = self._generate_samples(point_count)
            valid_mask = self._check_constraints(candidates)

            if np.all(valid_mask):
                # All points are valid
                return candidates
            else:
                if attempt == 0:
                    valid_count = np.sum(valid_mask)
                    _log.debug(
                        f"Initial design: {valid_count}/{point_count} points satisfy constraints. "
                        f"Retrying (attempt {attempt + 1}/{self.max_retries})."
                    )

        # Failed to generate valid samples after all retries
        raise RuntimeError(
            f"Could not generate {point_count} valid samples respecting all constraints "
            f"after {self.max_retries} attempts. "
            f"Consider relaxing constraints or increasing max_retries."
        )
