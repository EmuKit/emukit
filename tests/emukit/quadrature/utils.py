"""Helper functions for quadrature tests"""

from math import isclose

import numpy as np


def _compute_numerical_gradient(func, dfunc, in_shape):
    """Dimension that is being varied must be last dimension."""
    eps = 1e-8
    x = np.random.randn(*in_shape)
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


def check_grad(func, dfunc, in_shape):
    """``func`` must return ``np.ndarray`` of shape ``s`` and ``dfunc`` must return
    ``np.ndarray`` of shape ``s + (input_dim, )``."""
    ABS_TOL = 1e-4
    REL_TOL = 1e-5
    df, dft = _compute_numerical_gradient(func, dfunc, in_shape)
    isclose_all = np.array(
        [isclose(grad1, grad2, rel_tol=REL_TOL, abs_tol=ABS_TOL) for grad1, grad2 in zip(df.flatten(), dft.flatten())]
    )
    assert (1 - isclose_all).sum() == 0
