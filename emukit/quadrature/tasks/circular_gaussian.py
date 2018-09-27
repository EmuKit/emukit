import numpy as np
from scipy.integrate import dblquad
import  matplotlib.pyplot as plt

from .integrand import Integrand


class CircularGaussian(Integrand):
    """ 2D toy integrand that is a Gaussian on a circle """

    def __init__(self, mean, variance):
        """
        :param mean: mean of Gaussian in radius (must be > 0)
        """
        Integrand.__init__(self, input_dim=2)
        self.mean = mean
        self. variance = variance
        if self.mean < 0:
            raise ValueError('mean has to be > 0')

    def evaluate(self, x):
        """ Evaluate the function at x """
        if not isinstance(x, np.ndarray):
            raise ValueError('x must be an array with shape (N, 2)')
        norm_x = np.sqrt((x**2).sum(axis=1))
        return norm_x**2 * np.exp(- (norm_x - self.mean)**2/(2.*self.variance)) / np.sqrt(2. * np.pi * self.variance)

    def approximate_ground_truth_integral(self, lower_bounds, upper_bounds):
        """
        Use scipy.integrate.quad to estimate the ground truth

        :param lower_bounds: lower bounds of integral; np.ndarray
        :param upper_bounds: upper bounds of integral; np.ndarray

        :returns: output of scipy.integrate.quad
        """
        return dblquad(self._evaluate2d, lower_bounds[0], upper_bounds[0], lower_bounds[1], upper_bounds[1])

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

