# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.examples.fabolas import FabolasModel
from emukit.examples.fabolas.continuous_fidelity_entropy_search import ContinuousFidelityEntropySearch


def test_continuous_entropy_search():
    rng = np.random.RandomState(42)
    x_init = rng.rand(5, 1)
    s_min = 10
    s_max = 10000
    s = np.random.uniform(s_min, s_max, x_init.shape[0])
    x_init = np.concatenate((x_init, s[:, None]), axis=1)
    y_init = rng.rand(5, 1)

    model = FabolasModel(X_init=x_init, Y_init=y_init, s_min=s_min, s_max=s_max)

    space = ParameterSpace([ContinuousParameter("x", 0, 1), ContinuousParameter("s", np.log(s_min), np.log(s_max))])

    es = ContinuousFidelityEntropySearch(model, space, num_representer_points=10)
    es.update_pmin()

    assert np.all(es.representer_points[:, -1] == np.log(s_max))
