# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from emukit.experimental_design.acquisitions.integrated_variance import IntegratedVarianceReduction


def test_integrated_variance_fails_with_out_of_domain_test_points(gpy_model, continuous_space):
    """
    Checks that if the user supplies x_monte_carlo to the function
    and they are out of the domain of the parameter space a ValueError is raised.
    """
    x_monte_carlo = np.array([[0.5, 20.0], [0.2, 0.3]])

    with pytest.raises(ValueError):
        IntegratedVarianceReduction(gpy_model, continuous_space, x_monte_carlo)
