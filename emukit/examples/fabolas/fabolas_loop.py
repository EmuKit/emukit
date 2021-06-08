import numpy as np

from emukit.bayesian_optimization.loops.cost_sensitive_bayesian_optimization_loop import \
    CostSensitiveBayesianOptimizationLoop
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.acquisition import IntegratedHyperParameterAcquisition
from emukit.core.acquisition import acquisition_per_expected_cost
from emukit.core.loop import FixedIntervalUpdater, SequentialPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization import RandomSearchAcquisitionOptimizer
from emukit.examples.fabolas.continuous_fidelity_entropy_search import ContinuousFidelityEntropySearch
from emukit.examples.fabolas.fabolas_model import FabolasModel


class FabolasLoop(CostSensitiveBayesianOptimizationLoop):

    def __init__(self, space: ParameterSpace,
                 X_init: np.ndarray, Y_init: np.ndarray, cost_init: np.ndarray,
                 s_min: float, s_max: float,
                 update_interval: int = 1,
                 num_eval_points: int = 2000,
                 marginalize_hypers: bool = True):
        """
        Implements FAst Bayesian Optimization for LArge DataSets as described in:

        Fast Bayesian hyperparameter optimization on large datasets
        A. Klein and S. Falkner and S. Bartels and P. Hennig and F. Hutter
        Electronic Journal of Statistics (2017)

        :param space: input space where the optimization is carried out.
        :param X_init: initial data points
        :param Y_init: initial function values
        :param cost_init: initial costs
        :param s_min: smallest possible dataset size
        :param s_max: highest possible dataset size
        :param update_interval:  number of iterations between optimization of model hyper-parameters. Defaults to 1.
        :param num_eval_points: number of points to evaluate the acquisition function
        :param marginalize_hypers: if true, marginalize over the GP hyperparameters
        """

        l = space.parameters
        l.extend([ContinuousParameter("s", s_min, s_max)])  
        extended_space = ParameterSpace(l)

        model_objective = FabolasModel(X_init=X_init, Y_init=Y_init, s_min=s_min, s_max=s_max)
        model_cost = FabolasModel(X_init=X_init, Y_init=cost_init[:, None], s_min=s_min, s_max=s_max)

        if marginalize_hypers:
            acquisition_generator = lambda model: ContinuousFidelityEntropySearch(model_objective, space=extended_space,
                                                                                  target_fidelity_index=len(
                                                                                      extended_space.parameters) - 1)
            entropy_search = IntegratedHyperParameterAcquisition(model_objective, acquisition_generator)
        else:
            entropy_search = ContinuousFidelityEntropySearch(model_objective, space=extended_space,
                                                             target_fidelity_index=len(extended_space.parameters) - 1)

        acquisition = acquisition_per_expected_cost(entropy_search, model_cost)

        model_updater_objective = FixedIntervalUpdater(model_objective, update_interval)
        model_updater_cost = FixedIntervalUpdater(model_cost, update_interval, lambda state: state.cost)

        acquisition_optimizer = RandomSearchAcquisitionOptimizer(extended_space, num_eval_points=num_eval_points)
        candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)

        loop_state = create_loop_state(model_objective.X, model_objective.Y, cost=model_cost.Y)

        super(CostSensitiveBayesianOptimizationLoop, self).__init__(candidate_point_calculator,
                                                                    [model_updater_objective, model_updater_cost],
                                                                    loop_state)
