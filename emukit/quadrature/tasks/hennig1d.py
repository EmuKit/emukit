import numpy as np
from scipy.integrate import quad
import  matplotlib.pyplot as plt

from .integrand import Integrand


class Hennig1D(Integrand):
    """ 1D toy integrand coined by Philipp Hennig """

    def __init__(self):
        Integrand.__init__(self, input_dim=1)

    def evaluate(self, x):
        """ Evaluate the function at x """
        return np.exp(- x**2 - np.sin(3.*x)**2)

    def approximate_ground_truth_integral(self, lower_bounds, upper_bounds):
        """
        Use scipy.integrate.quad to estimate the ground truth

        :param lower_bounds: lower bounds of integral
        :param upper_bounds: upper bounds of integral

        :returns: output of scipy.integrate.quad
        """
        return quad(self.evaluate, lower_bounds, upper_bounds)

    def plot(self, resolution, lower_bounds, upper_bounds, **kwargs):
        x = np.linspace(lower_bounds, upper_bounds, resolution)
        plt.plot(x, self.evaluate(x), **kwargs)
        return