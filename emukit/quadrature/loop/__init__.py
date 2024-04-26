# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""Loops for Bayesian quadrature."""

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from .bayesian_monte_carlo_loop import BayesianMonteCarlo  # noqa: F401
from .bq_loop_state import QuadratureLoopState
from .bq_outer_loop import QuadratureOuterLoop
from .bq_stopping_conditions import CoefficientOfVariationStoppingCondition
from .vanilla_bq_loop import VanillaBayesianQuadratureLoop  # noqa: F401
from .wsabil_loop import WSABILLoop  # noqa: F401

__all__ = [
    "QuadratureOuterLoop",
    "BayesianMonteCarlo",
    "VanillaBayesianQuadratureLoop",
    "WSABILLoop",
    "QuadratureLoopState",
    "point_calculators",
    "CoefficientOfVariationStoppingCondition",
]
