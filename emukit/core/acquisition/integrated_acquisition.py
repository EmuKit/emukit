from typing import Union, Callable, Tuple

import numpy as np

from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel, IPriorHyperparameters


class IntegratedHyperParameterAcquisition(Acquisition):
    """
    This acquisition class provides functionality for integrating any acquisition function over model hyper-parameters
    """
    def __init__(self, model: Union[IModel, IPriorHyperparameters], acquisition_generator: Callable, n_samples: int=10,
                 n_burnin: int=100, subsample_interval: int=10, step_size: float=1e-1, leapfrog_steps: int=20):
        """
        :param model: An emukit model that implements IPriorHyperparameters
        :param acquisition_generator: Function that returns acquisition object when given the model as the only argument
        :param n_samples: Number of hyper-parameter samples
        :param n_burnin: Number of initial samples not used.
        :param subsample_interval: Interval of subsampling from HMC samples.
        :param step_size: Size of the gradient steps in the HMC sampler.
        :param leapfrog_steps: Number of gradient steps before each Metropolis Hasting step.
        """
        self.model = model
        self.acquisition_generator = acquisition_generator
        self.n_samples = n_samples
        self.n_burnin = n_burnin
        self.subsample_interval = subsample_interval
        self.step_size = step_size
        self.leapfrog_steps = leapfrog_steps

        self.update_parameters()

        acquisition = self.acquisition_generator(model)
        self._has_gradients = acquisition.has_gradients

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

        return acquisition_value / self.n_samples

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the acquisition value and its derivative integrating over the hyper-parameters of the model

        :param x: locations where the evaluation with gradients is done.
        :return: tuple containing the integrated expected improvement at the points x and its gradient.
        """

        if x.ndim == 1:
            x = x[None, :]

        acquisition_value = 0
        d_acquisition_dx = 0

        for sample in self.samples:
            self.model.fix_model_hyperparameters(sample)
            acquisition = self.acquisition_generator(self.model)
            improvement_sample, d_improvement_dx_sample = acquisition.evaluate_with_gradients(x)
            acquisition_value += improvement_sample
            d_acquisition_dx += d_improvement_dx_sample

        return acquisition_value / self.n_samples, d_acquisition_dx / self.n_samples

    def update_parameters(self):
        self.samples = self.model.generate_hyperparameters_samples(self.n_samples, self.n_burnin,
                                                                   self.subsample_interval,
                                                                   self.step_size, self.leapfrog_steps)

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return self._has_gradients

    def update_batches(self, x_batch, lipschitz_constant, f_min):
        acquisition = self.acquisition_generator(self.model)
        acquisition.update_batches(x_batch, lipschitz_constant, f_min)
