# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import abc
from typing import Tuple

import numpy as np


class Acquisition(abc.ABC):
    """ Acquisition base class """
    def __add__(self, other: 'Acquisition') -> 'Sum':
        """
        Overloads self + other
        """
        # If both acquisitions implement gradients, the gradients can be available in the sum
        return Sum(self, other)

    def __mul__(self, other: 'Acquisition') -> 'Product':
        """
        Overloads self * other
        """
        return Product(self, other)

    def __rmul__(self, other: 'Acquisition') -> 'Product':
        """
        Overloads other * self
        """
        return Product(other, self)

    def __truediv__(self, denominator: 'Acquisition') -> 'Quotient':
        """
        Overloads self / other
        """
        return Quotient(self, denominator)

    @abc.abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Abstract method. Evaluates the acquisition function.

        :param x: (n_points x n_dims) array of points at which to calculate acquisition function values
        :return: (n_points x 1) array of acquisition function values
        """
        pass

    @property
    @abc.abstractmethod
    def has_gradients(self) -> bool:
        """
        Abstract property. Whether acquisition value has analytical gradient calculation available.

        :return: True if gradients are available
        """
        pass

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optional abstract method that must be implemented if has_gradients returns True.
        Evaluates value and gradient of acquisition function at x.

        :param x: (n_points x n_dims) array of points at which to calculate acquisition function values and gradient
        :return: Tuple contains an (n_points x 1) array of acquisition function values and (n_points x n_dims) array of
                 acquisition function gradients with respect to x
        """
        raise NotImplementedError("Gradients not implemented for this acquisition function")

    def update_parameters(self) -> None:
        """
        Performs any updates to parameters that needs to be done once per outer loop iteration
        """
        pass


class Quotient(Acquisition):
    """
    Acquisition for division of two acquisition functions
    """
    def __init__(self, numerator: Acquisition, denominator: Acquisition):
        """

        :param numerator: Acquisition function to act as numerator in quotient
        :param denominator: Acquisition function to act as denominator in quotient
        """
        self.numerator = numerator
        self.denominator = denominator

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate division of the two acquisition functions

        :param x: (n_points x n_dims) array of points at which to calculate acquisition function values
        :return: (n_points x 1) array of acquisition function values
        """
        return self.numerator.evaluate(x) / self.denominator.evaluate(x)

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate value and gradient of acquisition function at x

        :param x: (n_points x n_dims) array of points at which to calculate acquisition function values and gradient
        :return: Tuple contains an (n_points x 1) array of acquisition function values and (n_points x n_dims) array of
                 acquisition function gradients with respect to x
        """
        # Evaluate both acquisition functions
        numerator_value, numerator_gradients = self.numerator.evaluate_with_gradients(x)
        denominator_value, denominator_gradients = self.denominator.evaluate_with_gradients(x)

        value = numerator_value / denominator_value
        # Calculate gradient of acquisition
        gradient = (numerator_gradients / denominator_value) \
                   - ((denominator_gradients * numerator_value) / (denominator_value**2))
        return value, gradient

    @property
    def has_gradients(self) -> bool:
        """
        Whether acquisition value has analytical gradient calculation available.

        :return: True if gradients are available
        """
        return self.denominator.has_gradients and self.numerator.has_gradients

    def update_parameters(self) -> None:
        """
        Performs any updates to parameters that needs to be done once per outer loop iteration
        """
        self.denominator.update_parameters()
        self.numerator.update_parameters()


class Product(Acquisition):
    """
    Acquisition for product of two or more acquisition functions
    """
    def __init__(self, acquisition_1: Acquisition, acquisition_2: Acquisition):
        """

        :param acquisition_1: Acquisition function in product
        :param acquisition_2: Other acquisition function in product
        """
        self.acquisition_1 = acquisition_1
        self.acquisition_2 = acquisition_2

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate product of the two acquisition functions

        :param x: (n_points x n_dims) array of points at which to calculate acquisition function values
        :return: (n_points x 1) array of acquisition function values
        """
        return self.acquisition_1.evaluate(x) * self.acquisition_2.evaluate(x)

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate value and gradient of acquisition function at x

        :param x: (n_points x n_dims) array of points at which to calculate acquisition function values and gradient
        :return: Tuple contains an (n_points x 1) array of acquisition function values and (n_points x n_dims) array of
                 acquisition function gradients with respect to x
        """
        # Evaluate acquisitions
        value_1, grad_1 = self.acquisition_1.evaluate_with_gradients(x)
        value_2, grad_2 = self.acquisition_2.evaluate_with_gradients(x)

        # Calculate gradient
        grad_total = value_1 * grad_2 + value_2 * grad_1

        return value_1 * value_2, grad_total

    @property
    def has_gradients(self):
        """
        Whether acquisition value has analytical gradient calculation available.

        :return: True if gradients are available
        """
        return self.acquisition_1.has_gradients and self.acquisition_2.has_gradients

    def update_parameters(self) -> None:
        """
        Performs any updates to parameters that needs to be done once per outer loop iteration
        """
        self.acquisition_1.update_parameters()
        self.acquisition_2.update_parameters()


class Sum(Acquisition):
    """
    Acquisition for sum of two acquisition functions
    """

    def __init__(self, acquisition_1: Acquisition, acquisition_2: Acquisition) -> None:
        """
        :param acquisition_1: An acquisition function in sum
        :param acquisition_2: Other acquisition function in sum
        """
        self.acquisition_1 = acquisition_1
        self.acquisition_2 = acquisition_2

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate sum of the two acquisition functions

        :param x: (n_points x n_dims) array of points at which to calculate acquisition function values
        :return: (n_points x 1) array of acquisition function values
        """
        return self.acquisition_1.evaluate(x) + self.acquisition_2.evaluate(x)

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate value and gradient of acquisition function at x

        :param x: (n_points x n_dims) array of points at which to calculate acquisition function values and gradient
        :return: Tuple contains an (n_points x 1) array of acquisition function values and (n_points x n_dims) array of
                 acquisition function gradients with respect to x
        """
        # Evaluate first acquisition with gradients
        value_1, grad_1 = self.acquisition_1.evaluate_with_gradients(x)
        value_2, grad_2 = self.acquisition_2.evaluate_with_gradients(x)

        return value_1 + value_2, grad_1 + grad_2

    @property
    def has_gradients(self):
        """
        Whether acquisition value has analytical gradient calculation available.

        :return: True if gradients are available
        """
        return self.acquisition_1.has_gradients and self.acquisition_2.has_gradients

    def update_parameters(self) -> None:
        """
        Performs any updates to parameters that needs to be done once per outer loop iteration
        """
        self.acquisition_1.update_parameters()
        self.acquisition_2.update_parameters()
