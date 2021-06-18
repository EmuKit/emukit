import numpy as np
from numpy.testing import assert_allclose
from collections import namedtuple
import pytest
from pytest_lazyfixture import lazy_fixture

from emukit.quadrature.methods.warpings import IdentityWarping, SquareRootWarping


def create_fixture_parameters():
    return [pytest.param(lazy_fixture(warping.name), id=warping.name) for warping in warpings]


@pytest.fixture
def identity_warping():
    return IdentityWarping()


@pytest.fixture
def squarerroot_warping():
    offset = 1.
    return SquareRootWarping(offset=offset)


@pytest.fixture
def inverted_squarerroot_warping():
    offset = 1.
    return SquareRootWarping(offset=offset, inverted=True)


warpings_tuple = namedtuple('WarpingTest', ['name'])
warpings = [warpings_tuple('identity_warping'),
            warpings_tuple('squarerroot_warping'),
            warpings_tuple('inverted_squarerroot_warping')
            ]


RTOL = 1e-8
ATOL = 1e-6


@pytest.mark.parametrize('warping', create_fixture_parameters())
def test_warping_shapes(warping):
    Y = np.ones([5, 1])
    assert warping.transform(Y).shape == Y.shape
    assert warping.inverse_transform(Y).shape == Y.shape


@pytest.mark.parametrize('warping', create_fixture_parameters())
def test_warping_values(warping):
    np.random.seed(42)
    Y = np.random.rand(5, 1)

    assert_allclose(warping.inverse_transform(warping.transform(Y)), Y, rtol=RTOL, atol=ATOL)


def test_squarerroot_warping_update_parameters(squarerroot_warping, inverted_squarerroot_warping):
    new_offset = 10.

    squarerroot_warping.update_parameters(offset=new_offset)
    assert squarerroot_warping.offset == new_offset

    inverted_squarerroot_warping.update_parameters(offset=new_offset)
    assert inverted_squarerroot_warping.offset == new_offset
