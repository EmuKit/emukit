from typing import Dict, Optional, Tuple

from GPyOpt.optimization import AcquisitionOptimizer
import numpy as np

from .acquisition_optimizer import AcquisitionOptimizerBase
from .context_manager import ContextManager
from .. import ParameterSpace
from ..acquisition import Acquisition

import logging
_log = logging.getLogger(__name__)


class GradientAcquisitionOptimizer(AcquisitionOptimizerBase):
    """ Optimizes the acquisition function using a quasi-Newton method (L-BFGS).
    Can be used for continuous acquisition functions.
    """
    def __init__(self, space: ParameterSpace, **kwargs) -> None:
        """
        :param space: The parameter space spanning the search problem.
        :param kwargs: Additional keyword arguments supported
                       by GPyOpt.optimization.AcquisitionOptimizer.
                       Note: only the 'lbfgs' optimizer is allowed.
        """
        super().__init__(space)

        if 'optimizer' in kwargs and kwargs['optimizer'] != 'lbfgs':
            raise ValueError("GradientAcquisitionOptimizer only supports"
                             "GPyOpt\'s lbfgs optimizer, got {}".format(kwargs['optimizer']))
        self.gpyopt_acquisition_optimizer = AcquisitionOptimizer(self.gpyopt_space, **kwargs)

    def _optimize(self, acquisition: Acquisition, context_manager: ContextManager)\
        -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of abstract method.
        Taking into account gradients if acquisition supports them.

        See AcquisitionOptimizerBase._optimizer for parameter descriptions.
        See class docstring for implementation details.
        """
        self.gpyopt_acquisition_optimizer.context_manager = context_manager._gpyopt_context_manager

        # Take negative of acquisition function because they are to be maximised and the optimizers minimise
        f = lambda x: -acquisition.evaluate(x)

        # Context validation
        if len(context_manager.contextfree_space.parameters) == 0:
            _log.warning("All parameters are fixed through context")
            x = np.array(context_manager._gpyopt_context_manager.context_value)[None, :]
            return x, f(x)

        def f_df(x):
            f_value, df_value = acquisition.evaluate_with_gradients(x)
            return -f_value, -df_value

        if acquisition.has_gradients:
            _log.info("Starting gradient-based optimization of acquisition function {}".format(type(acquisition)))
            x, f_min = self.gpyopt_acquisition_optimizer.optimize(f, None, f_df)
        else:
            _log.info("Starting gradient-free optimization of acquisition function {}".format(type(acquisition)))
            x, f_min = self.gpyopt_acquisition_optimizer.optimize(f, None, None)
        return x, -f_min
