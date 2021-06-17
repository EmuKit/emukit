import numpy as np
from typing import Optional


class Warping:
    """The warping for warped Bayesian quadrature."""

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from base-GP to integrand.

        :param Y: function values of latent function, shape (num_points, 1).
        :return: transformed values, shape (num_points, 1)
        """
        raise NotImplemented

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from integrand to base-GP.

        :param Y: function values of integrand, shape (num_points, 1).
        :return: transformed values, shape (num_points, 1)
        """
        raise NotImplemented

    def update_parameters(self, **kargs) -> None:
        """Update the warping parameters. Use `pass` if there are no parameters."""
        raise NotImplementedError


class IdentityWarping(Warping):

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from base-GP to integrand.

        :param Y: function values of latent function, shape (num_points, 1).
        :return: transformed values, shape (num_points, 1)
        """
        return Y

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from integrand to base-GP.

        :param Y: function values of integrand, shape (num_points, 1).
        :return: transformed values, shape (num_points, 1)
        """
        return Y

    def update_parameters(self) -> None:
        """No update for the identity transform."""
        pass


class SquareRootWarping(Warping):
    """The square root warping"""

    def __init__(self, offset: float, inverted: Optional[bool]=False):
        """
        :param offset: the offset of the warping
        :param inverted: inverts the warping if ``True``. Default is ``False``.
        """
        self.offset = offset
        self.inverted = inverted

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from base-GP to integrand.

        :param Y: function values of latent function, shape (num_points, 1).
        :return: transformed values, shape (num_points, 1)
        """
        if self.inverted:
            return self.offset - 0.5 * (Y * Y)
        else:
            return self.offset + 0.5 * (Y * Y)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Transform from integrand to base-GP.

        :param Y: function values of integrand, shape (num_points, 1).
        :return: transformed values, shape (num_points, 1)
        """
        if self.inverted:
            return np.sqrt(2. * (self.offset - Y))
        else:
            return np.sqrt(2. * (Y - self.offset))

    def update_parameters(self, offset: Optional[float]=None) -> None:
        """Update the :attr:`self.offset` if parameter is given.

        :param offset: the new value of :attr:`self.offset`
        """
        if offset is not None:
            self.offset = offset
