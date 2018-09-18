import GPyOpt

from .. import ParameterSpace
from ..acquisition import Acquisition


class AcquisitionOptimizer(object):
    """ Optimizes the acquisition function """
    def __init__(self, space: ParameterSpace, **kwargs) -> None:
        self.gpyopt_space = space.convert_to_gpyopt_design_space()
        self.gpyopt_acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(self.gpyopt_space, **kwargs)

    def optimize(self, acquisition: Acquisition):
        """
        acquisition - The acquisition function to be optimized
        """

        # Take negative of acquisition function because they are to be maximised and the optimizers minimise
        f = lambda x: -acquisition.evaluate(x)

        def f_df(x):
            f_value, df_value = acquisition.evaluate_with_gradients(x)
            return -f_value, -df_value

        if acquisition.has_gradients():
            return self.gpyopt_acquisition_optimizer.optimize(f, None, f_df)
        else:
            return self.gpyopt_acquisition_optimizer.optimize(f, None, None)
