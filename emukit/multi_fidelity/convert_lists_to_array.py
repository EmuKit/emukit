# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
This module provides functions to convert from a list representation of multi-fidelity data to an array representation.

The list based representation is a list of numpy arrays, with a numpy array for every fidelity. The list is ordered
from the lowest fidelity to the highest fidelity.

The array representation is one array for all inputs where the last column of the X array is a zero-based index
indicating the fidelity.
"""
from typing import List, Tuple

import numpy as np


def convert_x_list_to_array(x_list: List) -> np.ndarray:
    """
    Converts list representation of features to array representation
    :param x_list: A list of (n_points x n_dims) numpy arrays ordered from lowest to highest fidelity
    :return: An array of all features with the zero-based fidelity index appended as the last column
    """
    # First check everything is a 2d array
    if not np.all([x.ndim == 2 for x in x_list]):
        raise ValueError('All x arrays must have 2 dimensions')

    x_array = np.concatenate(x_list, axis=0)
    indices = []
    for i, x in enumerate(x_list):
        indices.append(i * np.ones((len(x), 1)))

    x_with_index = np.concatenate((x_array, np.concatenate(indices)), axis=1)
    return x_with_index


def convert_y_list_to_array(y_list: List) -> np.ndarray:
    """
    Converts list representation of outputs to array representation
    :param y_list: A list of (n_points x n_outputs) numpy arrays representing the outputs
                   ordered from lowest to highest fidelity
    :return: An array of all outputs
     """
    if not np.all([y.ndim == 2 for y in y_list]):
        raise ValueError('All y arrays must have 2 dimensions')
    return np.concatenate(y_list, axis=0)


def convert_xy_lists_to_arrays(x_list: List, y_list: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts list representation of targets to array representation
    :param x_list: A list of (n_points x n_dims) numpy arrays ordered from lowest to highest fidelity
    :param y_list: A list of (n_points x n_outputs) numpy arrays representing the outputs
                   ordered from lowest to highest fidelity
    :return: Tuple of (x_array, y_array) where
             x_array contains all inputs across all fidelities with the fidelity index appended as the last column
             and y_array contains all outputs across all fidelities.
    """

    if len(x_list) != len(y_list):
        raise ValueError('Different number of fidelities between x and y')

    # Check same number of points in each fidelity
    n_points_x = np.array([x.shape[0] for x in x_list])
    n_points_y = np.array([y.shape[0] for y in y_list])
    if not np.all(n_points_x == n_points_y):
        raise ValueError('Different number of points in x and y at the same fidelity')

    return convert_x_list_to_array(x_list), convert_y_list_to_array(y_list)
