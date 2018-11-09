

import numpy as np
from scipy.integrate import quad, dblquad
import  matplotlib.pyplot as plt

from emukit.core import MultiDimensionalContinuousParameter


class Hennig1D():
    """ 1D toy integrand coined by Philipp Hennig """

    def evaluate(self, x):
        """ Evaluate the function at x """
        return np.exp(- x**2 - np.sin(3.*x)**2)

    def approximate_ground_truth_integral(self, integral_bounds: MultiDimensionalContinuousParameter):
        """
        Use scipy.integrate.quad to estimate the ground truth

        :param integral_bounds: bounds of integral

        :returns: output of scipy.integrate.quad
        """
        lb, ub = integral_bounds.get_bounds()
        return quad(self.evaluate, np.float(lb), np.float(ub))

    def plot(self, resolution, lower_bounds, upper_bounds, **kwargs):
        x = np.linspace(lower_bounds, upper_bounds, resolution)
        plt.plot(x, self.evaluate(x), **kwargs)
        return


class CircularGaussian():
    """ 2D toy integrand that is a Gaussian on a circle """

    def __init__(self, mean: np.float, variance: np.float):
        """
        :param mean: mean of Gaussian in radius (must be > 0)
        """
        Integrand.__init__(self, input_dim=2)
        self.mean = mean
        self. variance = variance

    def evaluate(self, x):
        """ Evaluate the function at x """
        norm_x = np.sqrt((x**2).sum(axis=1))
        return norm_x**2 * np.exp(- (norm_x - self.mean)**2/(2.*self.variance)) / np.sqrt(2. * np.pi * self.variance)

    def approximate_ground_truth_integral(self, integral_bounds: MultiDimensionalContinuousParameter):
        """
        Use scipy.integrate.quad to estimate the ground truth

        :param integral_bounds: bounds of integral

        :returns: output of scipy.integrate.quad
        """
        lb, ub = integral_bounds.get_bounds()
        return dblquad(self._evaluate2d, lb[0], ub[0], lb[1], ub[1])

    def plot(self, resolution, lower_bounds, upper_bounds, **kwargs):
        """ Surface plot of the integrand """
        X1, X2 = np.meshgrid(np.linspace(lower_bounds[0], upper_bounds[0], resolution),
                             np.linspace(lower_bounds[1], upper_bounds[1], resolution), indexing='ij')
        x = np.concatenate((X1.reshape(-1, 1), X2.reshape(-1, 1)), axis=1)

        f = plt.figure(figsize=(5,5))
        ax = f.add_subplot(111)
        ax.contourf(X1, X2, self.evaluate(x).reshape(resolution, resolution), cmap=plt.cm.Blues)
        ax.axis([lower_bounds[0], upper_bounds[0], lower_bounds[1], upper_bounds[1]])
        ax.axis('equal')
        return

    # helpers
    def _evaluate2d(self, y, x):
        """ dblquad doesn't take np.ndarrays as input, but two floats"""
        return self.evaluate(np.array([x, y]).reshape(1, -1)).item()

