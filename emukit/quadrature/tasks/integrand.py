import abc

class Integrand(abc.ABC):
    """
    Abstract class for toy integrands
    """

    def __init__(self, input_dim):
        """
        :param input_dim: Input dimension of the integrand
        """
        self.input_dim = input_dim

    @abc.abstractmethod
    def evaluate(self, x):
        """
        Evaluate the function at x

        :param x: Argument of the integrand
        """
        pass

    @abc.abstractmethod
    def approximate_ground_truth_integral(self, lower_bounds, upper_bounds): # TODO this only works in 1 and 2d, shouldn't be here
        """
        Use scipy.integrate to estimate the ground truth

        :param lower_bounds: lower bounds of integral
        :param upper_bounds: upper bounds of integral

        :returns: numerical solution to the integral, absolute error
        """
        pass

    @abc.abstractmethod
    def plot(self, *args, **kwargs): # TODO also only 1 or 2d
        """ plot the integrand """
        pass


# TODO new abstract class Integrand_with_ground_truth(Integrand): that contains true integral