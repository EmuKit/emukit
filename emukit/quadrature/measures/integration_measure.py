# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from ...core.optimization.context_manager import ContextManager
from ..typing import BoundsType


class IntegrationMeasure:
    """An abstract class for an integration measure defined by a density.

    :param name: Name of the integration measure

    """

    def __init__(self, name: str):
        self.name = name

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the density at x.

        :param x: Points at which density is evaluated, shape (n_points, input_dim).
        :return: The density at x, shape (n_points, ).
        """
        raise NotImplementedError

    def compute_density_gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the gradient of the density at x.

        :param x: Points at which the gradient is evaluated, shape (n_points, input_dim).
        :return: The gradient of the density at x, shape (n_points, input_dim).
        """
        raise NotImplementedError

    def get_box(self) -> BoundsType:
        """A meaningful box containing the measure.

        Outside this box, the measure should be zero or virtually zero.

        :return: The meaningful box.
        """
        raise NotImplementedError

    @property
    def can_sample(self) -> bool:
        """Indicates whether the measure has sampling available.

        :return: ``True`` if sampling is available. ``False`` otherwise.
        """
        raise NotImplementedError

    def get_samples(self, num_samples: int, context_manager: ContextManager = None) -> np.ndarray:
        """Samples from the measure.

        :param num_samples: The number of samples to be taken.
        :param context_manager: The context manager that contains variables to fix and the values to fix them to.
                                If a context is given, this method samples from the conditional distribution.
        :return: The samples, shape (num_samples, input_dim).
        """
        raise NotImplementedError
