try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError('pytorch is not installed. Please installed version it by running pip install torch torchvision')

try:
    from pybnn.util.layers import AppendLayer
except ImportError:
    raise ImportError('pybnn is not installed. Please installed version it by running pip install pybnn')


def get_default_architecture_classification(input_dimensionality: int) -> torch.nn.Module:
    """
    Defined the architecture that is uses for meta-classification benchmarks (i.e meta-svm and meta-fcnet),
    compared to regression it only outputs mean prediction values in [0, 1].

    :param input_dimensionality: dimensionality of the benchmark
    :return: nn.Module
    """
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=500):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, 1)
            self.sigma_layer = AppendLayer(noise=1e-3)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
            mean = torch.sigmoid(x)
            return self.sigma_layer(mean)

    return Architecture(n_inputs=input_dimensionality)


def get_default_architecture_regression(input_dimensionality: int) -> torch.nn.Module:
    """
    Defined the architecture that is uses for meta-regression benchmarks (i.e meta-xgboost)

    :param input_dimensionality: dimensionality of the benchmark
    :return: nn.Module
    """
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=500):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, 1)
            self.sigma_layer = AppendLayer(noise=1e-3)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
            mean = x
            return self.sigma_layer(mean)

    return Architecture(n_inputs=input_dimensionality)


def get_default_architecture_cost(input_dimensionality: int) -> torch.nn.Module:
    """
    Defined the architecture that is uses to model the costs.

    :param input_dimensionality: dimensionality of the benchmark
    :return: nn.Module
    """
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=500):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, 1)
            self.sigma_layer = AppendLayer(noise=1e-3)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
            mean = x
            return self.sigma_layer(mean)

    return Architecture(n_inputs=input_dimensionality)
