import pickle

try:
    import torch
except ImportError:
    raise ImportError('pytorch is not installed. Please install it by running pip install torch torchvision')


from functools import partial
from typing import Tuple

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.examples.profet.meta_benchmarks.architecture import get_default_architecture
from emukit.examples.profet.meta_benchmarks.meta_surrogates import objective_function


def meta_xgboost(fname_objective: str, fname_cost: str, noise: bool=True) -> Tuple[UserFunctionWrapper, ParameterSpace]:
    """
    Interface to the Meta-XGBoost benchmark which imitates the hyperparameter optimization of
    XGBoost on UCI like regression datasets.
    Offline generated function samples can be download here:

    http://www.ml4aad.org/wp-content/uploads/2019/05/profet_data.tar.gz

    NOTE: make sure that the index for the objective function and the cost function match,
    e.g for sample_objective_i.pkl and sample_cost_i.pkl the index i should be the same.
    Each of those files contain the parameters of a neural network which represents a single function sample and
    some additional information, such as the mean and variance of the input normalization.

    For further information about Profet and the generated meta-surrogate benchmarks see:

    Meta-Surrogate Benchmarking for Hyperparameter Optimization
    A. Klein and Z. Dai and F. Hutter and N. Lawrence and J. Gonzalez
    arXiv:1905.12982 [cs.LG] (2019)

    :param fname_objective: filename for the objective function
    :param fname_cost: filename for the cost function
    :param noise: determines whether to add noise on the function value or not
    :return: Tuple of user function object and parameter space
    """
    parameter_space = ParameterSpace([
        ContinuousParameter('learning_rate', 0, 1),  # original space was in [1e-6, 1e-1]
        ContinuousParameter('gamma', 0, 1),  # original space was in [0, 2]
        ContinuousParameter('reg_alpha', 0, 1),  # original space was in [1e-5, 1e3]
        ContinuousParameter('reg_lambda', 0, 1),  # original space was in [1e-4, 1e3]
        ContinuousParameter('n_estimators', 0, 1),  # original space was in [10, 500]
        ContinuousParameter('subsample', 0, 1),  # original space was in [1e-1, 1]
        ContinuousParameter('max_depth', 0, 1),  # original space was in [1, 15]
        ContinuousParameter('min_child_weight', 0, 1)])  # original space was in [0, 20]

    data = pickle.load(open(fname_objective, "rb"))

    x_mean_objective = data["x_mean"]
    x_std_objective = data["x_std"]
    y_mean_objective = data["y_mean"]
    y_std_objective = data["y_std"]
    task_feature_objective = data["task_feature"]
    objective = get_default_architecture(x_mean_objective.shape[0]).float()
    objective.load_state_dict(data["state_dict"])

    data = pickle.load(open(fname_cost, "rb"))

    x_mean_cost = data["x_mean"]
    x_std_cost = data["x_std"]
    y_mean_cost = data["y_mean"]
    y_std_cost = data["y_std"]
    task_feature_cost = data["task_feature"]
    cost = get_default_architecture(x_mean_cost.shape[0]).float()
    cost.load_state_dict(data["state_dict"])

    f = partial(objective_function, model_objective=objective, model_cost=cost,
                task_feature_objective=task_feature_objective, task_feature_cost=task_feature_cost,
                x_mean_objective=x_mean_objective, x_std_objective=x_std_objective,
                x_mean_cost=x_mean_cost, x_std_cost=x_std_cost,
                y_mean_objective=y_mean_objective, y_std_objective=y_std_objective,
                y_mean_cost=y_mean_cost, y_std_cost=y_std_cost,
                log_objective=True, with_noise=noise)

    return f, parameter_space
