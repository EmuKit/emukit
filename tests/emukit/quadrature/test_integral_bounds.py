# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from emukit.quadrature.kernels.integral_bounds import IntegralBounds


def test_integral_bounds_values():
    bounds = [(-1, 1), (-2, 0)]
    lower_bounds = np.array([[-1, -2]])
    upper_bounds = np.array([[1, 0]])

    bounds = IntegralBounds(name='test_name', bounds=bounds)
    res = bounds.get_lower_and_upper_bounds()
    assert len(res) == 2
    assert np.all(res[0] == lower_bounds)
    assert np.all(res[1] == upper_bounds)

    assert len(bounds.convert_to_list_of_continuous_parameters()) == 2
    assert bounds.name == 'test_name'


def test_integral_bounds_wrong_bounds():
    bounds_wrong = [(-1, 1), (0, -2)]

    with pytest.raises(ValueError):
        IntegralBounds(name='test_name', bounds=bounds_wrong)
