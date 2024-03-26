# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.examples.profet.performance_assessment import compute_ecdf, compute_ranks, compute_runtime_feval


def test_compute_runtime_feval():
    traj = np.arange(0, 10)[::-1]
    rt = compute_runtime_feval(traj, 5)

    assert rt == 5


def test_compute_ecdf():
    error = np.random.randn(100, 20, 50)
    targets = np.random.rand(100, 10)

    error_range, cdf = compute_ecdf(error, targets)

    assert len(error_range) == len(cdf)


def compute_ranks():
    error = np.random.rand(2, 10, 10, 20)
    ranks = compute_ranks(error)

    assert ranks.shape[0] == 2
    assert ranks.shape[1] == 20
