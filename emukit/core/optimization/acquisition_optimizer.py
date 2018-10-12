from typing import Tuple

import GPyOpt
import numpy as np

from .. import ParameterSpace
from ..acquisition import Acquisition


import logging
_log = logging.getLogger(__name__)

class AcquisitionOptimizer(object):
    """ Optimizes the acquisition function """
    def __init__(self, space: ParameterSpace, **kwargs) -> None:
        self.gpyopt_space = space.convert_to_gpyopt_design_space()
        self.gpyopt_acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(self.gpyopt_space, **kwargs)

    def optimize(self, acquisition: Acquisition, context: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimizes the acquisition function, taking into account gradients if it supports them

        :param acquisition: The acquisition function to be optimized
        :param context: Optimization context, determines whether any variable values should be fixed during the optimization
        """

        self.gpyopt_acquisition_optimizer.context_manager = GPyOpt.optimization.acquisition_optimizer.ContextManager(
            self.gpyopt_space, context)

        # Take negative of acquisition function because they are to be maximised and the optimizers minimise
        f = lambda x: -acquisition.evaluate(x)

        def f_df(x):
            f_value, df_value = acquisition.evaluate_with_gradients(x)
            return -f_value, -df_value

        if acquisition.has_gradients:
            _log.info("Starting gradient-based optimization of acquisition function {}".format(type(acquisition)))
            return self.gpyopt_acquisition_optimizer.optimize(f, None, f_df)
        else:
            _log.info("Starting gradient-free optimization of acquisition function {}".format(type(acquisition)))
            return self.gpyopt_acquisition_optimizer.optimize(f, None, None)
