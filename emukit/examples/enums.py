from enum import Enum


class AcquisitionType(Enum):
    EI = 1
    PI = 2
    NLCB = 3


class ModelType(Enum):
    RandomForest = 1
    BayesianNeuralNetwork = 2
