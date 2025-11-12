# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0



# Importing multi_fidelity_kernel should succeed without GPy. Accessing GPy-specific classes
# should raise an informative ImportError
# pointing users to install the optional extra: `pip install emukit[gpy]`.
# By doing it this way, we avoid breaking minimal installs of Emukit.
from importlib import util as _importlib_util

if _importlib_util.find_spec("GPy") is not None:  # GPy available
    from .linear_multi_fidelity_kernel import LinearMultiFidelityKernel  # noqa: F401
else:
    class LinearMultiFidelityKernel:  # pragma: no cover - exercised only when GPy missing
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GPy is not installed. Install optional dependency with 'pip install emukit[gpy]' to use LinearMultiFidelityKernel."
            )

__all__ = ["LinearMultiFidelityKernel"]
