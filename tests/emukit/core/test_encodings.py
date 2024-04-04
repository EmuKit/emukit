# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from numpy.testing import assert_array_equal

from emukit.core import OneHotEncoding, OrdinalEncoding


@pytest.fixture
def categories():
    return ["one", "two", "three"]


def test_one_hot_encoding(categories):
    encoding = OneHotEncoding(categories)

    assert encoding.dimension == 3
    assert encoding.get_category([1, 0, 0]) == "one"
    assert encoding.get_encoding("three") == [0, 0, 1]

    with pytest.raises(ValueError):
        encoding.get_category([1, 1, 0])

    with pytest.raises(ValueError):
        encoding.get_encoding("four")


def test_one_hot_encoding_rounding(categories):
    encoding = OneHotEncoding(categories)

    check_rounding(encoding, np.array([[1, 0, 0]]), np.array([[1, 0, 0]]))
    check_rounding(encoding, np.array([[0.5, 0.1, 0.1]]), np.array([[1, 0, 0]]))
    check_rounding(encoding, np.array([[2, 0, 1.5]]), np.array([[1, 0, 0]]))
    check_rounding(encoding, np.array([[0.7, 0.75, 0.5]]), np.array([[0, 1, 0]]))

    with pytest.raises(ValueError):
        encoding.round(np.array([1, 0, 0]))

    with pytest.raises(ValueError):
        encoding.round(np.array([[1, 0]]))


def test_ordinal_encoding(categories):
    encoding = OrdinalEncoding(categories)

    assert encoding.dimension == 1
    assert encoding.get_category([1]) == "one"
    assert encoding.get_encoding("three") == [3]


def test_ordinal_encoding_rounding(categories):
    encoding = OrdinalEncoding(categories)

    check_rounding(encoding, np.array([[1]]), np.array([[1]]))
    check_rounding(encoding, np.array([[0.5]]), np.array([[1]]))
    check_rounding(encoding, np.array([[2]]), np.array([[2]]))
    check_rounding(encoding, np.array([[5]]), np.array([[3]]))

    with pytest.raises(ValueError):
        encoding.round(np.array([1]))

    with pytest.raises(ValueError):
        encoding.round(np.array([[1, 1]]))


def check_rounding(encoding, x, expected):
    rounded_x = encoding.round(x)
    assert_array_equal(expected, rounded_x)
