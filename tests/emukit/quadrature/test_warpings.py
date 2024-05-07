# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from collections import namedtuple

import numpy as np
import pytest
from numpy.testing import assert_allclose

from emukit.quadrature.methods.warpings import IdentityWarping, SquareRootWarping


@pytest.fixture
def identity_warping():
    return IdentityWarping()


@pytest.fixture
def square_root_warping():
    offset = 1.0
    return SquareRootWarping(offset=offset)


@pytest.fixture
def inverted_square_root_warping():
    offset = 1.0
    return SquareRootWarping(offset=offset, is_inverted=True)


warpings = [
    "identity_warping",
    "square_root_warping",
    "inverted_square_root_warping",
]


RTOL = 1e-8
ATOL = 1e-6


@pytest.mark.parametrize("warping_name", warpings)
def test_warping_shapes(warping_name, request):
    warping = request.getfixturevalue(warping_name)
    Y = np.ones([5, 1])
    assert warping.transform(Y).shape == Y.shape
    assert warping.inverse_transform(Y).shape == Y.shape


@pytest.mark.parametrize("warping_name", warpings)
def test_warping_values(warping_name, request):
    warping = request.getfixturevalue(warping_name)
    np.random.seed(42)
    Y = np.random.rand(5, 1)

    assert_allclose(warping.inverse_transform(warping.transform(Y)), Y, rtol=RTOL, atol=ATOL)


def test_square_root_warping_update_parameters(square_root_warping, inverted_square_root_warping):
    new_offset = 10.0

    square_root_warping.update_parameters(offset=new_offset)
    assert square_root_warping.offset == new_offset

    inverted_square_root_warping.update_parameters(offset=new_offset)
    assert inverted_square_root_warping.offset == new_offset


def test_square_root_warping_inverted_flag(square_root_warping, inverted_square_root_warping):
    assert not square_root_warping.is_inverted
    assert inverted_square_root_warping.is_inverted
