import pickle
import torch
import torch.nn as nn
import numpy as np

from functools import partial
from typing import Tuple

from pybnn.util.layers import AppendLayer

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.loop.user_function import UserFunctionWrapper


def get_architecture_forrester(input_dimensionality: int) -> torch.nn.Module:
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=100):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs, n_hidden)
            self.fc2 = nn.Linear(n_hidden, 2)
            self.sigma_layer = AppendLayer(noise=1e-3)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = self.fc2(x)
            mean = x[:, None, 0]
            return self.sigma_layer(mean)

    return Architecture(n_inputs=input_dimensionality)


def meta_forrester(fname_objective: str) -> Tuple[UserFunctionWrapper, ParameterSpace]:
    """

    Interface to the MetaForrester benchmark described in:

    Meta-Surrogate Benchmarking for Hyperparameter Optimization
    A. Klein and Z. Dai and F. Hutter and N. Lawrence and J. Gonzalez
    arXiv:1905.12982 [cs.LG] (2019)

    :param fname_objective: filename for the objective function
    :return: Tuple of user function object and parameter space
    """
    parameter_space = ParameterSpace([
        ContinuousParameter('x', 0, 1)])
    data = pickle.load(open(fname_objective, "rb"))
    x_mean_objective = data["x_mean"]
    x_std_objective = data["x_std"]
    y_mean_objective = data["y_mean"]
    y_std_objective = data["y_std"]
    task_feature_objective = data["task_feature"]

    objective = get_architecture_forrester(x_mean_objective.shape[0]).float()
    objective.load_state_dict(data["state_dict"])

    def objective_function(config):

        Ht = np.repeat(task_feature_objective[None, :], config.shape[0], axis=0)
        x = np.concatenate((config, Ht), axis=1)
        x_norm = torch.from_numpy((x - x_mean_objective) / x_std_objective).float()
        o = objective.forward(x_norm).data.numpy()
        m = o[:, 0]

        feval = m * y_std_objective + y_mean_objective

        return feval[:, None]

    f = partial(objective_function)

    return f, parameter_space
