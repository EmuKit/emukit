import pytest
import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.sensitivity.test_functions import Ishigami

def ishigami_function():

    ishigami = Ishigami(a=5, b=0.1)

    assert(ishigami.fidelity1(np.array([[0,1,0]]))[0] == 3.5403670913678558)
    assert(ishigami.fidelity2(np.array([[0,1,0]]))[0] == 4.2073549240394827)
    assert(ishigami.fidelity3(np.array([[0,1,0]]))[0] == 3.363348736799463)
    assert(ishigami.fidelity4(np.array([[0,1,0]]))[0] == 2.1242202548207136)

    assert(ishigami.f0() == 2.5)
    assert(iishigami.f1(np.array([[1]]))[0,0] == 2.48080946)
    assert(iishigami.f2(np.array([[1]]))[0,0] == 1.0403670913678558)
    assert(iishigami.f3(np.array([[1]]))[0,0] == 0)
    assert(iishigami.f12(np.array([[1,1]]))[0] == 0)
    assert(iishigami.f13(np.array([[1,1]]))[0] == -1.5551913767516916)
    assert(iishigami.f23(np.array([[1,1]]))[0] == 0)
    assert(iishigami.f123(np.array([[1,1]]))[0] == 0)
