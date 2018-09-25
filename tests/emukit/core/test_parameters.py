import numpy as np
from emukit.core import OrdinalParameter

def test_ordinal_parameter():
    param = OrdinalParameter('x', [0, 1, 2])
    assert param.check_in_domain(np.array([0])) is True
    assert param.check_in_domain(np.array([3])) is False