# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.test_functions.sensitivity import Ishigami


def test_ishigami_function():
    ishigami = Ishigami(a=5, b=0.1)

    assert ishigami.fidelity1(np.array([[0, 1, 0]])).shape == (1,)
    assert ishigami.fidelity2(np.array([[0, 1, 0]])).shape == (1,)
    assert ishigami.fidelity3(np.array([[0, 1, 0]])).shape == (1,)
    assert ishigami.fidelity4(np.array([[0, 1, 0]])).shape == (1,)

    assert ishigami.f1(np.array([[1]])).shape == (1, 1)
    assert ishigami.f2(np.array([[1]])).shape == (1, 1)
    assert ishigami.f3(np.array([[1]])).shape == (1,)
    assert ishigami.f12(np.array([[1, 1]])).shape == (1,)
    assert ishigami.f13(np.array([[1, 1]])).shape == (1,)
    assert ishigami.f23(np.array([[1, 1]])).shape == (1,)
    assert ishigami.f123(np.array([[1, 1]])).shape == (1,)
