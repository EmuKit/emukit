# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from numpy.testing import assert_array_equal

from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace
from emukit.core.optimization import GradientAcquisitionOptimizer, MultiSourceAcquisitionOptimizer


def test_multi_source_acquisition_optimizer(simple_square_acquisition):
    space = ParameterSpace([ContinuousParameter("x", 0, 1), InformationSourceParameter(2)])
    single_optimizer = GradientAcquisitionOptimizer(space)
    optimizer = MultiSourceAcquisitionOptimizer(single_optimizer, space)

    opt_x, opt_val = optimizer.optimize(simple_square_acquisition)
    assert_array_equal(opt_x, np.array([[0.0, 1.0]]))
    assert_array_equal(opt_val, np.array([[2.0]]))
