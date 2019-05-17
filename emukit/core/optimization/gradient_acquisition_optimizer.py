import logging
from typing import Tuple

import numpy as np

from .. import ParameterSpace
from ..acquisition import Acquisition
from .acquisition_optimizer import AcquisitionOptimizerBase
from .anchor_points_generator import ObjectiveAnchorPointsGenerator
from .context_manager import ContextManager
from .optimizer import apply_optimizer, OptLbfgs

_log = logging.getLogger(__name__)


class GradientAcquisitionOptimizer(AcquisitionOptimizerBase):
    """ Optimizes the acquisition function using a quasi-Newton method (L-BFGS).
    Can be used for continuous acquisition functions.
    """
    def __init__(self, space: ParameterSpace) -> None:
        """
        :param space: The parameter space spanning the search problem.
        :param kwargs: Additional keyword arguments supported by GPyOpt.optimization.AcquisitionOptimizer.
                       Note: only the 'lbfgs' optimizer is allowed.
        """
        super().__init__(space)

    def _optimize(self, acquisition: Acquisition, context_manager: ContextManager) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of abstract method.
        Taking into account gradients if acquisition supports them.

        See AcquisitionOptimizerBase._optimizer for parameter descriptions.
        See class docstring for implementation details.
        """

        # Take negative of acquisition function because they are to be maximised and the optimizers minimise
        f = lambda x: -acquisition.evaluate(x)

        # Context validation
        if len(context_manager.contextfree_space.parameters) == 0:
            _log.warning("All parameters are fixed through context")
            x = np.array(context_manager.context_values)[None, :]
            return x, f(x)

        if acquisition.has_gradients:
            def f_df(x):
                f_value, df_value = acquisition.evaluate_with_gradients(x)
                return -f_value, -df_value
        else:
            f_df = None

        optimizer = OptLbfgs(context_manager.contextfree_space.get_bounds())

        anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, acquisition)

        # Select the anchor points (with context)
        anchor_points = anchor_points_generator.get(num_anchor=1, context_manager=context_manager)

        _log.info("Starting gradient-based optimization of acquisition function {}".format(type(acquisition)))
        optimized_points = []
        for a in anchor_points:
            optimized_point = apply_optimizer(optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=None,
                                              context_manager=context_manager, space=self.space)
            optimized_points.append(optimized_point)

        x_min, fx_min = min(optimized_points, key=lambda t: t[1])
        return x_min, -fx_min
