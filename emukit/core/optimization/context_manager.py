from typing import Any, Dict, Optional

import numpy as np
from GPyOpt.optimization.acquisition_optimizer import ContextManager as GPyOptContextManager

from .. import ParameterSpace


Context = Dict[str, Any]


class ContextManager:
    """
    Handles the context variables in the optimizer
    """
    def __init__(self, space: ParameterSpace,
                 context: Context,
                 gpyopt_space: Optional[Dict[str, Any]] = None):
        """
        :param space: Parameter space of the search problem.
        :param context: Dictionary of variables and their context values.
                        These values are fixed while optimization.
        :param gpyopt_space: Same as space but in GPyOpt format.
        """
        self.space = space
        if gpyopt_space is None:
            gpyopt_space = space.convert_to_gpyopt_design_space()
        self._gpyopt_context_manager = GPyOptContextManager(gpyopt_space, context)
        self.contextfree_space = ParameterSpace(
            [param for param in self.space.parameters if param.name not in context])
        self.context_space = ParameterSpace(
            [param for param in self.space.parameters if param.name in context])

    def expand_vector(self, x: np.ndarray) -> np.ndarray:
        """
        Expand contextfree parameter vector by values of the context.

        :param x: Contextfree parameter values as 2d-array
        :return: Parameter values with inserted context values
        """
        if len(self.context_space.parameters) == 0:
            return x
        else:
            return self._gpyopt_context_manager._expand_vector(x)
