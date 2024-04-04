# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""Acquisition functions for the quadrature package."""

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from .mutual_information import MutualInformation  # noqa: F401
from .squared_correlation import SquaredCorrelation  # noqa: F401
from .squared_correlation import SquaredCorrelation as IntegralVarianceReduction
from .uncertainty_sampling import UncertaintySampling  # noqa: F401

__all__ = [
    "MutualInformation",
    "IntegralVarianceReduction",
    "SquaredCorrelation",
    "UncertaintySampling",
]
