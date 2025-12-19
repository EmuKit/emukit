# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# GPy-based model classes should raise an informative ImportError pointing users to
# install the optional extra: `pip install emukit[gpy]`.

from importlib import util as _importlib_util

# Importing multi_fidelity_kernel should succeed without GPy. Accessing GPy-specific classes
# should raise an informative ImportError
# pointing users to install the optional extra: `pip install emukit[gpy]`.
# By doing it this way, we avoid breaking minimal installs of Emukit.
if _importlib_util.find_spec("GPy") is not None:  # GPy available
    from .linear_model import GPyLinearMultiFidelityModel  # noqa: F401
    from .non_linear_multi_fidelity_model import NonLinearMultiFidelityModel  # noqa: F401
else:
    class _GPyMissingBase:  # pragma: no cover - exercised in minimal installs
        _error_msg = (
            "GPy is not installed. Install optional dependency with 'pip install emukit[gpy]' "
            "to use {name}."
        )

        def __init__(self, *args, **kwargs):  # pragma: no cover
            raise ImportError(self._error_msg.format(name=self.__class__.__name__))

    class GPyLinearMultiFidelityModel(_GPyMissingBase):  # pragma: no cover
        """Placeholder for GPyLinearMultiFidelityModel. Requires `emukit[gpy]` extra."""

    class NonLinearMultiFidelityModel(_GPyMissingBase):  # pragma: no cover
        """Placeholder for NonLinearMultiFidelityModel. Requires `emukit[gpy]` extra."""

__all__ = ["GPyLinearMultiFidelityModel", "NonLinearMultiFidelityModel"]
