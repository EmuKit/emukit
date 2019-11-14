try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError('pytorch is not installed. Please install it by running pip install torch torchvision')

try:
    from pybnn.util.layers import AppendLayer
except ImportError:
    raise ImportError('pybnn is not installed. Please install it by running pip install pybnn')


def get_default_architecture(input_dimensionality: int,
                             classification: bool = False,
                             n_hidden: int = 500) -> torch.nn.Module:
    """
    Defines the architecture that is used for Meta-Surrogate benchmarks.
    In the case of emulating a classification benchmark, we pass the mean prediction through a sigmoid
    to make sure that the mean prediction values are in [0, 1].

    :param input_dimensionality: dimensionality of the benchmark
    :param classification: defined whether we emulate a classification benchmark
    :param n_hidden: number of units in the hidden layer
    :return: nn.Module
    """
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=500, classification=False):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, 1)
            self.sigma_layer = AppendLayer(noise=1e-3)
            self.classification = classification

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
            if self.classification:
                mean = torch.sigmoid(x)
            else:
                mean = x
            return self.sigma_layer(mean)

    return Architecture(n_inputs=input_dimensionality,
                        n_hidden=n_hidden,
                        classification=classification)
