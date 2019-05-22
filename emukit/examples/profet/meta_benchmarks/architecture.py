import torch
import torch.nn as nn
from pybnn.util.layers import AppendLayer


def get_default_architecture_classification(input_dimensionality: int) -> torch.nn.Module:
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=500):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, 2)
            self.sigma_layer = AppendLayer(noise=1e-3)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
            mean = torch.sigmoid(x[:, None, 0])
            return self.sigma_layer(mean)

    return Architecture(n_inputs=input_dimensionality)


def get_default_architecture_regression(input_dimensionality: int) -> torch.nn.Module:
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=500):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, 2)
            self.sigma_layer = AppendLayer(noise=1e-3)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
            mean = x[:, None, 0]
            return self.sigma_layer(mean)

    return Architecture(n_inputs=input_dimensionality)


def get_default_architecture_cost(input_dimensionality: int) -> torch.nn.Module:
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=500):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, 2)
            self.sigma_layer = AppendLayer(noise=1e-3)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
            mean = x[:, None, 0]
            return self.sigma_layer(mean)

    return Architecture(n_inputs=input_dimensionality)
