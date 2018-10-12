# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Contains functions that are useful to showcase multi-fidelity models
"""

import numpy as np


def forrester_high(x):
    """
    High fidelity version of Forrester function

    Reference:
    Engineering design via surrogate modelling: a practical guide.
    Forrester, A., Sobester, A., & Keane, A. (2008).
    """
    return (6 * x[:, 0] - 2)**2 * np.sin(12 * x[:, 0] - 4)


def forrester_low(x):
    """
    Low fidelity version of Forrester function

    Reference:
    Engineering design via surrogate modelling: a practical guide.
    Forrester, A., Sobester, A., & Keane, A. (2008).
    """
    return 0.5 * forrester_high(x) + 10 * (x[:, 0] - 0.5) + 5


def borehole_high(x):
    """
    High fidelity version of borehole function

    The Borehole function models water flow through a borehole.
    Its simplicity and quick evaluation makes it a commonly used
    function for testing a wide variety of methods in computer experiments.

    Reference:
    https://www.sfu.ca/~ssurjano/borehole.html
    """

    numerator = 2 * np.pi * x[:, 2] * (x[:, 3] - x[:, 5])
    ln_r_rw = np.log(x[:, 1] / x[:, 0])
    denominator = ln_r_rw * \
        (x[:, 2] / x[:, 4] + 1 + (2 * x[:, 6] * x[:, 2]) /
         (x[:, 0]**2 * x[:, 7] * ln_r_rw))
    return numerator / denominator


def borehole_low(x):
    """
    Low fidelity version of borehole function

    The Borehole function models water flow through a borehole.
    Its simplicity and quick evaluation makes it a commonly used
    function for testing a wide variety of methods in computer experiments.

    Reference:
    https://www.sfu.ca/~ssurjano/borehole.html
    """

    numerator = 5 * x[:, 2] * (x[:, 3] - x[:, 5])
    ln_r_rw = np.log(x[:, 1] / x[:, 0])
    denominator = ln_r_rw * \
        (x[:, 2] / x[:, 4] + 1.5 + (2 * x[:, 6] * x[:, 2]) /
         (x[:, 0]**2 * x[:, 7] * ln_r_rw))
    return numerator / denominator

def nonlinear_sin_low(x):
    """
    Low fidelity version of nonlinear sin function

    Reference:
    Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling. 
    P. Perdikaris, M. Raissi, A. Damianou, N. D. Lawrence and G. E. Karniadakis (2017)
    http://web.mit.edu/parisp/www/assets/20160751.full.pdf
    """

    return np.sin(8 * np.pi * x)

def nonlinear_sin_high(x):
    """
    High fidelity version of nonlinear sin function

    Reference:
    Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling. 
    P. Perdikaris, M. Raissi, A. Damianou, N. D. Lawrence and G. E. Karniadakis (2017)
    http://web.mit.edu/parisp/www/assets/20160751.full.pdf
    """
    
    return (x - np.sqrt(2)) * nonlinear_sin_low(x)**2