import pickle
import torch
import numpy as np

from functools import partial
from typing import Tuple

from emukit.core import ContinuousParameter, ParameterSpace, DiscreteParameter
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.examples.profet.meta_benchmarks.architecture import get_default_architecture_classification, \
    get_default_architecture_cost


def meta_fcnet(fname_objective: str, fname_cost: str, noise: bool=True) -> Tuple[UserFunctionWrapper, ParameterSpace]:
    """

    Interface to the MetaFCNet benchmark described in:

    Meta-Surrogate Benchmarking for Hyperparameter Optimization

    :param fname_objective: filename for the objective function
    :param fname_cost: filename for the cost function
    :param noise: determines whether to add noise on the function value or not
    :return: Tuple of user function object and parameter space
    """
    parameter_space = ParameterSpace([
        ContinuousParameter('lr', 1e-6, 1e-1),
        DiscreteParameter('batch_size', 8, 128),
        DiscreteParameter('n_units_1', 16, 512),
        DiscreteParameter('n_units_2', 16, 512),
        ContinuousParameter('dropout_1', 0, 0.99),
        ContinuousParameter('dropout_2', 0, 0.99),
    ])
    data = pickle.load(open(fname_objective, "rb"))

    x_mean_objective = data["x_mean"]
    x_std_objective = data["x_std"]
    y_mean_objective = data["y_mean"]
    y_std_objective = data["y_std"]
    task_feature_objective = data["task_feature"]
    objective = get_default_architecture_classification(x_mean_objective.shape[0]).float()
    objective.load_state_dict(data["state_dict"])

    data = pickle.load(open(fname_cost, "rb"))

    x_mean_cost = data["x_mean"]
    x_std_cost = data["x_std"]
    y_mean_cost = data["y_mean"]
    y_std_cost = data["y_std"]
    task_feature_cost = data["task_feature"]
    cost = get_default_architecture_cost(x_mean_cost.shape[0]).float()
    cost.load_state_dict(data["state_dict"])

    def objective_function(config, with_noise=True):

        Ht = np.repeat(task_feature_objective[None, :], config.shape[0], axis=0)
        x = np.concatenate((config, Ht), axis=1)
        x_norm = torch.from_numpy((x - x_mean_objective) / x_std_objective).float()
        o = objective.forward(x_norm).data.numpy()
        m = o[:, 0] * y_std_objective + y_mean_objective
        log_v = o[:, 1] * y_std_objective ** 2
        if with_noise:
            feval = np.random.randn() * np.sqrt(np.exp(log_v)) + m
        else:
            feval = m

        Ht = np.repeat(task_feature_cost[None, :], config.shape[0], axis=0)
        x = np.concatenate((config, Ht), axis=1)
        x_norm = torch.from_numpy((x - x_mean_cost) / x_std_cost).float()
        o = cost.forward(x_norm).data.numpy()
        log_m = o[:, 0] * y_std_cost + y_mean_cost
        log_log_v = o[:, 1] * y_std_cost ** 2
        if with_noise:
            log_c = np.random.randn() * np.sqrt(np.exp(log_log_v)) + log_m
        else:
            log_c = log_m

        return feval[:, None], np.exp(log_c)[:, None]

    f = partial(objective_function, with_noise=noise)
    # user_function = UserFunctionWrapper(f)
    return f, parameter_space
