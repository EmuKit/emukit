import abc
import numpy as np
from typing import Optional


class Warping(abc.ABC):
    """The warping for warped Bayesian quadrature."""

    @abc.abstractmethod
    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from base-GP to integrand.

        :param Y: Function values of latent function, shape (n_points, 1).
        :return: Transformed values, shape (n_points, 1).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from integrand to base-GP.

        :param Y: Function values of integrand, shape (n_points, 1).
        :return: Transformed values, shape (n_points, 1).
        """
        raise NotImplementedError

    def update_parameters(self, **new_parameters) -> None:
        """Update the warping parameters. The keyword arguments ``new_parameters`` contain the parameter names as
        keys with the new values. An empty dictionary will not update any parameters."""
        for parameter, new_value in new_parameters.items():
            setattr(self, parameter, new_value)


class IdentityWarping(Warping):

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from base-GP to integrand.

        :param Y: Function values of latent function, shape (n_points, 1).
        :return: Transformed values, shape (n_points, 1).
        """
        return Y

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from integrand to base-GP.

        :param Y: Function values of integrand, shape (n_points, 1).
        :return: Transformed values, shape (n_points, 1).
        """
        return Y


class SquareRootWarping(Warping):
    """The square root warping"""

    def __init__(self, offset: float, is_inverted: Optional[bool]=False):
        """
        :param offset: The offset of the warping.
        :param is_inverted: Inverts the warping if ``True``. Default is ``False``.
        """
        self.offset = offset
        self.is_inverted = is_inverted

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from base-GP to integrand.

        :param Y: Function values of latent function, shape (n_points, 1).
        :return: Transformed values, shape (n_points, 1).
        """
        if self.is_inverted:
            return self.offset - 0.5 * (Y * Y)
        else:
            return self.offset + 0.5 * (Y * Y)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from integrand to base-GP.

        :param Y: Function values of integrand, shape (n_points, 1).
        :return: Transformed values, shape (n_points, 1).
        """
        if self.is_inverted:
            return np.sqrt(2. * (self.offset - Y))
        else:
            return np.sqrt(2. * (Y - self.offset))
