import numpy as np

from emukit.core import ParameterSpace
from emukit.core import ContinuousParameter, InformationSourceParameter
from emukit.core.acquisition import Acquisition
from emukit.core.optimization import AcquisitionOptimizer
from emukit.core.optimization import MultiSourceAcquisitionOptimizer


class SimpleSquareAcquisition(Acquisition):
    def __init__(self):
        pass

    def evaluate(self, x):
        y = - x[:, 0]**2 + np.sum(x[:, 1:], axis=1) + 1
        return np.atleast_2d(y).T

    @property
    def has_gradients(self):
        return False


def test_acquisition_optimizer():
    space = ParameterSpace([ContinuousParameter('x', 0, 1)])
    acquisition = SimpleSquareAcquisition()
    optimizer = AcquisitionOptimizer(space)

    opt_x, opt_val = optimizer.optimize(acquisition)
    np.testing.assert_array_equal(opt_x, np.array([[0.]]))
    np.testing.assert_array_equal(opt_val, np.array([[1.]]))


def test_multi_source_acquisition_optimizer():
    space = ParameterSpace([ContinuousParameter('x', 0, 1),
                            InformationSourceParameter(2)])
    acquisition = SimpleSquareAcquisition()
    single_optimizer = AcquisitionOptimizer(space)
    optimizer = MultiSourceAcquisitionOptimizer(single_optimizer, space)

    opt_x, opt_val = optimizer.optimize(acquisition)
    np.testing.assert_array_equal(opt_x, np.array([[0., 1.]]))
    np.testing.assert_array_equal(opt_val, np.array([[2.]]))
