# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import numpy as np

from .. import ParameterSpace

Context = Dict[str, Any]


class ContextManager:
    """
    Handles the context variables in the optimizer
    """
    def __init__(self, space: ParameterSpace, context: Context):
        """
        :param space: Parameter space of the search problem.
        :param context: Dictionary of variables and their context values.
                        These values are fixed while optimization.
        """
        self.space = space
        self.contextfree_space = ParameterSpace(
            [param for param in self.space.parameters if param.name not in context])
        self.context_space = ParameterSpace(
            [param for param in self.space.parameters if param.name in context])

        # Find indices of context and non context variables
        self.context_idxs = []
        self.context_values = []
        for name in context.keys():
            # Find indices of variable in the input domain
            self.context_idxs += self.space.find_parameter_index_in_model(name)

            # Find encoded values of context variable
            param = self.space.get_parameter_by_name(name)
            if hasattr(param, 'encoding'):
                self.context_values.append(param.encoding.get_encoding(context[name]))
            else:
                self.context_values.append(context[name])

        all_idxs = list(range(space.dimensionality))
        self.non_context_idxs = [idx for idx in all_idxs if idx not in self.context_idxs]

    def expand_vector(self, x: np.ndarray) -> np.ndarray:
        """
        Expand context free parameter vector by values of the context.

        :param x: Context free parameter values as 2d-array
        :return: Parameter values with inserted context values
        """
        if len(self.context_space.parameters) == 0:
            return x
        else:
            x = np.atleast_2d(x)
            x_expanded = np.zeros((x.shape[0], self.space.dimensionality))
            x_expanded[:, np.array(self.non_context_idxs).astype(int)] = x
            x_expanded[:, np.array(self.context_idxs).astype(int)] = self.context_values
            return x_expanded
