# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""Helper functions for quadrature tests"""

from math import isclose
from typing import Callable, Tuple

import numpy as np

from emukit.quadrature.typing import BoundsType


def _compute_numerical_gradient(
    func: Callable, dfunc: Callable, in_shape: Tuple[int, ...], bounds: BoundsType
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the numerical gradient of ``func``.

    :param func: Function must take inputs of shape ``in_shape = s + (input_dim, )`` and return np.ndarray of shape ``s``.
    :param dfunc: Gradient of function to be tested. Must return np.ndarray of shape ``(input_dim) + s``.
    :param in_shape: The input shape to ``func``. Must be of form ``in_shape = s + (input_dim, )``.
    :param bounds: The bounds for the x values.
    :return: The gradient ``dfunc`` evaluated at a random location and its numerical gradient.
    """
    eps = 1e-8
    x = sample_uniform(in_shape, bounds)
    f = func(x)
    df = dfunc(x)
    dft = np.zeros(df.shape)
    for d in range(x.shape[-1]):
        x_tmp = x.copy()
        x_tmp[..., d] = x_tmp[..., d] + eps
        f_tmp = func(x_tmp)
        dft_d = (f_tmp - f) / eps
        dft[d, ...] = dft_d
    return df, dft


def check_grad(
    func: Callable,
    dfunc: Callable,
    in_shape: Tuple[int, ...],
    bounds: BoundsType,
    atol: float = 1e-4,
    rtol: float = 1e-5,
) -> None:
    """Asserts if the gradient of a function is close to its numerical gradient.

    :param func: Function must take inputs of shape ``in_shape = s + (input_dim, )`` and return np.ndarray of shape ``s``.
    :param dfunc: Gradient of function to be tested. Must return np.ndarray of shape ``(input_dim) + s``.
    :param in_shape: The input shape to ``func``. Must be of form ``in_shape = s + (input_dim, )``.
    :param bounds: The bounds for the x values.
    :param atol: Absolute tolerance of the closeness check. Defaults to 1e-4.
    :param atol: Relative tolerance of the closeness check. Defaults to 1e-5.
    """
    df, dft = _compute_numerical_gradient(func, dfunc, in_shape, bounds)
    isclose_all = np.array(
        [isclose(grad1, grad2, rel_tol=rtol, abs_tol=atol) for grad1, grad2 in zip(df.flatten(), dft.flatten())]
    )
    assert (1 - isclose_all).sum() == 0


def sample_uniform(in_shape, bounds):
    n_samples = np.prod(in_shape[:-1])
    samples = np.zeros(in_shape)
    for i, b in enumerate(bounds):
        samples[..., i] = np.random.uniform(low=b[0], high=b[1], size=n_samples)
    return samples
