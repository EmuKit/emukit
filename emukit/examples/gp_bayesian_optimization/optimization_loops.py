import numpy as np

from ...bayesian_optimization.acquisitions import ExpectedImprovement, ProbabilityOfImprovement, \
    NegativeLowerConfidenceBound
from ...core import ParameterSpace
from ...core.loop import OuterLoop, FixedIntervalUpdater, SequentialPointCalculator
from ...core.loop.loop_state import create_loop_state
from ...core.optimization import AcquisitionOptimizer
from ..models.bohamiann import Bohamiann
from ..models.random_forest import RandomForest
from .enums import AcquisitionType, ModelType


def create_bayesian_optimization_loop(x_init: np.ndarray, y_init: np.ndarray, parameter_space: ParameterSpace,
                                      acquisition_type: AcquisitionType, model_type: ModelType,
                                      model_kwargs: dict=None) -> OuterLoop:
    """
    Creates Bayesian optimization loop for Bayesian neural network or random forest models.

    :param x_init: 2d numpy array of shape (no. points x no. input features) of initial X data
    :param y_init: 2d numpy array of shape (no. points x no. targets) of initial Y data
    :param parameter_space: parameter space describing input domain
    :param acquisition_type: an AcquisitionType enumeration object describing which acquisition function to use
    :param model_type: A ModelType enumeration object describing which model to use.
    :param model_kwargs: Key work arguments for the model constructor. See individual models for details.
    :return: OuterLoop instance
    """

    if model_kwargs is None:
        model_kwargs = dict()

    # Create model
    if model_type is ModelType.RandomForest:
        model = RandomForest(x_init, y_init, **model_kwargs)
    elif model_type is ModelType.BayesianNeuralNetwork:
        model = Bohamiann(x_init, y_init, **model_kwargs)
    else:
        raise ValueError('Unrecognised model type: ' + str(model_type))

    # Create acquisition
    if acquisition_type is AcquisitionType.EI:
        acquisition = ExpectedImprovement(model)
    elif acquisition_type is AcquisitionType.PI:
        acquisition = ProbabilityOfImprovement(model)
    elif acquisition_type is AcquisitionType.NLCB:
        acquisition = NegativeLowerConfidenceBound(model)
    else:
        raise ValueError('Unrecognised acquisition type: ' + str(acquisition_type))

    acquisition_optimizer = AcquisitionOptimizer(parameter_space)
    candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)
    loop_state = create_loop_state(x_init, y_init)
    model_updater = FixedIntervalUpdater(model, 1)
    return OuterLoop(candidate_point_calculator, model_updater, loop_state)
