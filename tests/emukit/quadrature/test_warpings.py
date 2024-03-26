# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from collections import namedtuple

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytest_lazyfixture import lazy_fixture

from emukit.quadrature.methods.warpings import IdentityWarping, SquareRootWarping


def create_fixture_parameters():
    return [pytest.param(lazy_fixture(warping.name), id=warping.name) for warping in warpings]


@pytest.fixture
def identity_warping():
    return IdentityWarping()


@pytest.fixture
def squarerroot_warping():
    offset = 1.0
    return SquareRootWarping(offset=offset)


@pytest.fixture
def inverted_squarerroot_warping():
    offset = 1.0
    return SquareRootWarping(offset=offset, is_inverted=True)


warpings_tuple = namedtuple("WarpingTest", ["name"])
warpings = [
    warpings_tuple("identity_warping"),
    warpings_tuple("squarerroot_warping"),
    warpings_tuple("inverted_squarerroot_warping"),
]


RTOL = 1e-8
ATOL = 1e-6


@pytest.mark.parametrize("warping", create_fixture_parameters())
def test_warping_shapes(warping):
    Y = np.ones([5, 1])
    assert warping.transform(Y).shape == Y.shape
    assert warping.inverse_transform(Y).shape == Y.shape


@pytest.mark.parametrize("warping", create_fixture_parameters())
def test_warping_values(warping):
    np.random.seed(42)
    Y = np.random.rand(5, 1)

    assert_allclose(warping.inverse_transform(warping.transform(Y)), Y, rtol=RTOL, atol=ATOL)


def test_squarerroot_warping_update_parameters(squarerroot_warping, inverted_squarerroot_warping):
    new_offset = 10.0

    squarerroot_warping.update_parameters(offset=new_offset)
    assert squarerroot_warping.offset == new_offset

    inverted_squarerroot_warping.update_parameters(offset=new_offset)
    assert inverted_squarerroot_warping.offset == new_offset


def test_squarerroot_warping_inverted_flag(squarerroot_warping, inverted_squarerroot_warping):
    assert not squarerroot_warping.is_inverted
    assert inverted_squarerroot_warping.is_inverted
