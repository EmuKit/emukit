import numpy as np

from emukit.core import DiscreteParameter


def test_discrete_parameter():
    param = DiscreteParameter('x', [0, 1, 2])
    assert param.check_in_domain(np.array([0])) is True
    assert param.check_in_domain(np.array([3])) is False