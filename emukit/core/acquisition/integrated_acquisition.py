from typing import Union, Callable, Tuple

import numpy as np

from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel, IPriorHyperparameters, IDifferentiable


class IntegratedAcquisition(Acquisition):
    """
    This acquisition class provides functionality for integrating any other functionality over model hyper-parameters
    """
    def __init__(self, model: Union[IModel, IPriorHyperparameters], acquisition_generator: Callable, n_samples: int=10):
        """
        :param model: An emukit model that implements IPriorHyperparameters
        :param acquisition_generator: Function that returns acquisition object when given the model as the only argument
        :param n_samples: Number of hyper-parameter samples to integrate over
        """
        self.model = model
        self.acquisition_generator = acquisition_generator
        self.n_samples = n_samples
        self.samples = self.model.generate_hyperparameters_samples(n_samples)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate acquisition by integrating over the hyper-parameters of the model
        :param x: locations where the evaluation is done.
        :return: Array with integrated acquisition value at all input locations
        """
        acquisition_value = 0
        for sample in self.samples:
            self.model.fix_model_hyperparameters(sample)
            acquisition = self.acquisition_generator(self.model)
            acquisition_value += acquisition.evaluate(x)

        return acquisition_value/self.n_samples

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the acquisition value and its derivative integrating over the hyper-parameters of the model

        :param x: locations where the evaluation with gradients is done.
        :return: tuple containing numpy arrays with the integrated expected improvement at the points x
        and its gradient.
        """

        if x.ndim == 1: x = x[None, :]
        acquisition_value = 0
        dacquisition_dx = 0

        for sample in self.samples:
            self.model.fix_model_hyperparameters(sample)
            acquisition = self.acquisition_generator(self.model)
            improvement_sample, dimprovement_dx_sample = acquisition.evaluate_with_gradients(x)
            acquisition_value += improvement_sample
            dacquisition_dx += dimprovement_dx_sample

        return acquisition_value/self.n_samples, dacquisition_dx/self.n_samples

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return isinstance(self.model, IDifferentiable)
