import numpy as np

from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.loop.user_function import MultiSourceFunctionWrapper


def multi_fidelity_non_linear_sin(high_fidelity_noise_std_deviation=0, low_fidelity_noise_std_deviation=0):
    """
    Two level non-linear sin function where high fidelity is given by:

    .. math::
        f_{high}(x) = (x - \sqrt{2}) f_{low}(x)^2

    and the low fidelity is:

    .. math::
        f_{low}(x) = \sin(8 \pi x)

    Reference:
    Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling.
    P. Perdikaris, M. Raissi, A. Damianou, N. D. Lawrence and G. E. Karniadakis (2017)
    http://web.mit.edu/parisp/www/assets/20160751.full.pdf
    """

    parameter_space = ParameterSpace([ContinuousParameter('x1', -5, 10), InformationSourceParameter(2)])
    user_function = MultiSourceFunctionWrapper([
        lambda x: nonlinear_sin_low(x, low_fidelity_noise_std_deviation),
        lambda x: nonlinear_sin_high(x, high_fidelity_noise_std_deviation)])
    return user_function, parameter_space


def nonlinear_sin_low(x, sd=0):
    """
    Low fidelity version of nonlinear sin function
    """

    return np.sin(8 * np.pi * x) + np.random.randn(x.shape[0], 1) * sd


def nonlinear_sin_high(x, sd=0):
    """
    High fidelity version of nonlinear sin function
    """

    return (x - np.sqrt(2)) * nonlinear_sin_low(x, 0) ** 2 + np.random.randn(x.shape[0], 1) * sd
