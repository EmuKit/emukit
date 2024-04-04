# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.bayesian_optimization import epmgp


def test_joint_min():
    # Uniform distribution
    n_points = 5
    m = np.ones([n_points])
    v = np.eye(n_points)

    pmin = epmgp.joint_min(m, v)
    pmin = np.exp(pmin)

    uprob = 1.0 / n_points

    assert pmin.shape[0] == n_points
    assert np.any(pmin < (uprob + 0.03)) and np.any(pmin > uprob - 0.01)

    # Dirac delta
    m = np.ones([n_points]) * 1000
    m[0] = 1
    v = np.eye(n_points)

    pmin = epmgp.joint_min(m, v)
    pmin = np.exp(pmin)
    assert pmin[0] == 1.0
    assert np.any(pmin[:1] > 1e-10)
