# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import List


class Parameter(object):
    @property
    def model_dimension(self) -> int:
        return 1

    @property
    def model_params(self) -> List:
        return [self]

    def round(self, x) -> np.ndarray:
        return x

    def check_in_domain(self, x: np.ndarray) -> bool:
        """
        Verifies that given values lie within the parameter's domain

        :param x: Value to be checked
        :return: A boolean value which indicates whether all points lie in the domain
        """
        raise NotImplemented
