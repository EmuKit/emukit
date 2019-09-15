import pickle
import numpy as np

from functools import partial
from typing import Tuple

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError('pytorch is not installed. Please installed version it by running pip install torch torchvision')

try:
    from pybnn.util.layers import AppendLayer
except ImportError:
    raise ImportError('pybnn is not installed. Please install it by running pip install pybnn')

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
    Interface to the Meta-Forrester benchmark.

    Offline generated function samples can be download here:

    http://www.ml4aad.org/wp-content/uploads/2019/05/profet_data.tar.gz

    For further information about Profet and the generated meta-surrogate benchmarks see:

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
        output = objective.forward(x_norm).data.numpy()
        mean = output[:, 0]

        feval = mean * y_std_objective + y_mean_objective

        return feval[:, None]

    f = partial(objective_function)

    return f, parameter_space
