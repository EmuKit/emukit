import numpy as np
from typing import Optional


class Warping:
    """The warping for warped Bayesian quadrature."""

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from base-GP to integrand.

        :param Y: Function values of latent function, shape (n_points, 1).
        :return: Transformed values, shape (n_points, 1).
        """
        raise NotImplemented

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from integrand to base-GP.

        :param Y: Function values of integrand, shape (n_points, 1).
        :return: Transformed values, shape (n_points, 1).
        """
        raise NotImplemented

    def update_parameters(self, **kwargs) -> None:
        """Update the warping parameters."""
        pass


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

    def update_parameters(self, offset: Optional[float]=None) -> None:
        """Update the :attr:`self.offset` if parameter is given.

        :param offset: The new value of :attr:`self.offset`.
        """
        if offset is not None:
            self.offset = offset
