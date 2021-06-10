import numpy as np
from typing import Optional


class Warping:
    """The warping"""

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Transform from base-GP to integrand.
        """
        raise NotImplemented

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Transform from integrand to base-GP.
        """
        raise NotImplemented

    def update_parameters(self, **kargs) -> None:
        """update the warping parameters"""
        raise NotImplementedError


class IdentityWarping(Warping):

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """ Transform from base-GP to integrand """
        return Y

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """ Transform from integrand to base-GP """
        return Y

    def update_parameters(self) -> None:
        """update the warping parameters"""
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
        """ Transform from base-GP to integrand """
        if self.inverted:
            return self.offset - 0.5 * (Y * Y)
        else:
            return self.offset + 0.5 * (Y * Y)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """ Transform from integrand to base-GP """
        if self.inverted:
            return np.sqrt(np.clip(2. * (self.offset - Y), a_min=0., a_max=None))
        else:
            return np.sqrt(np.clip(2. * (Y - self.offset), a_min=0., a_max=None))

    def update_parameters(self, offset: Optional[float]=None) -> None:
        """update the warping parameters"""
        if offset is not None:
            self.offset = offset
