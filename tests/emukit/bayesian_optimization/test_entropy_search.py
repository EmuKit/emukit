# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from emukit.bayesian_optimization.acquisitions import EntropySearch
from emukit.samplers import AffineInvariantEnsembleSampler


@pytest.fixture
def entropy_search_acquisition(gpy_model, continuous_space):
    sampler = AffineInvariantEnsembleSampler(continuous_space)
    return EntropySearch(gpy_model, continuous_space, sampler, num_representer_points=10)


def test_entropy_search_update_pmin(entropy_search_acquisition):
    logP = entropy_search_acquisition.update_pmin()

    assert logP.shape[0] == entropy_search_acquisition.num_representer_points
    # Check if representer points are inside the bounds
    assert np.all(
        np.all(entropy_search_acquisition.representer_points > 0)
        & np.all(entropy_search_acquisition.representer_points < 1)
    )


def test_innovations(entropy_search_acquisition):
    # Case 1: Assume no influence of test point on representer points
    entropy_search_acquisition.representer_points = np.array([[100.0, 100.0]])
    x = np.array([[0.0, 0.0]])
    dm, dv = entropy_search_acquisition._innovations(x)

    assert np.any(np.abs(dm) < 1e-3)
    assert np.any(np.abs(dv) < 1e-3)

    # Case 2: Test point is close to representer points
    entropy_search_acquisition.representer_points = np.array([[1.0, 1.0]])
    x = np.array([[0.99, 0.99]])
    dm, dv = entropy_search_acquisition._innovations(x)
    assert np.any(np.abs(dm) > 1e-3)
    assert np.any(np.abs(dv) > 1e-3)
