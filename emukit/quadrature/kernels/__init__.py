# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""Kernel embeddings for Bayesian quadrature."""

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from .quadrature_kernels import (  # isort: skip
    GaussianEmbedding,
    LebesgueEmbedding,
    QuadratureKernel,
    QuadratureProductKernel,
)
from .quadrature_brownian import (
    QuadratureBrownian,
    QuadratureBrownianLebesgueMeasure,
    QuadratureProductBrownian,
    QuadratureProductBrownianLebesgueMeasure,
)
from .quadrature_matern12 import QuadratureProductMatern12, QuadratureProductMatern12LebesgueMeasure
from .quadrature_matern32 import QuadratureProductMatern32, QuadratureProductMatern32LebesgueMeasure
from .quadrature_matern52 import QuadratureProductMatern52, QuadratureProductMatern52LebesgueMeasure
from .quadrature_rbf import QuadratureRBF, QuadratureRBFGaussianMeasure, QuadratureRBFLebesgueMeasure

__all__ = [
    "QuadratureKernel",
    "QuadratureProductKernel",
    "LebesgueEmbedding",
    "GaussianEmbedding",
    "QuadratureBrownian",
    "QuadratureBrownianLebesgueMeasure",
    "QuadratureRBF",
    "QuadratureRBFLebesgueMeasure",
    "QuadratureRBFGaussianMeasure",
    "QuadratureProductMatern52",
    "QuadratureProductMatern52LebesgueMeasure",
    "QuadratureProductMatern32",
    "QuadratureProductMatern32LebesgueMeasure",
    "QuadratureProductMatern12",
    "QuadratureProductMatern12LebesgueMeasure",
    "QuadratureProductBrownian",
    "QuadratureProductBrownianLebesgueMeasure",
]
