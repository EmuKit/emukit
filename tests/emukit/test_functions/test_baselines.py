# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from emukit.test_functions.quadrature.baselines import (
    bivariate_approximate_ground_truth_integral,
    univariate_approximate_ground_truth_integral,
)


def test_univariate_approximate_ground_truth_integral_shape():
    """
    Test output dimension is 2d
    """
    res = univariate_approximate_ground_truth_integral(lambda x: 1., (0., 1.))
    assert len(res) == 2


def test_bivariate_approximate_ground_truth_integral_shape():
    """
    Test output dimension is 2d
    """
    res = bivariate_approximate_ground_truth_integral(lambda x: 1., 2*[(0., 1.)])
    assert len(res) == 2
