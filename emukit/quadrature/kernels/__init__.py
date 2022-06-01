"""Kernel embeddings for Bayesian quadrature."""

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from .quadrature_kernels import QuadratureKernel, QuadratureProductKernel  # isort:skip
from .quadrature_brownian import (
    QuadratureBrownian,
    QuadratureBrownianLebesgueMeasure,
    QuadratureProductBrownian,
    QuadratureProductBrownianLebesgueMeasure,
)
from .quadrature_matern32 import QuadratureProductMatern32, QuadratureProductMatern32LebesgueMeasure
from .quadrature_matern52 import QuadratureProductMatern52, QuadratureProductMatern52LebesgueMeasure
from .quadrature_rbf import (
    QuadratureRBF,
    QuadratureRBFIsoGaussMeasure,
    QuadratureRBFLebesgueMeasure,
    QuadratureRBFUniformMeasure,
)

__all__ = [
    "QuadratureKernel",
    "QuadratureBrownian",
    "QuadratureBrownianLebesgueMeasure",
    "QuadratureRBF",
    "QuadratureRBFLebesgueMeasure",
    "QuadratureRBFUniformMeasure",
    "QuadratureRBFIsoGaussMeasure",
    "QuadratureProductKernel",
    "QuadratureProductBrownian",
    "QuadratureProductBrownianLebesgueMeasure",
    "QuadratureProductMatern32",
    "QuadratureProductMatern32LebesgueMeasure",
    "QuadratureProductMatern52",
    "QuadratureProductMatern52LebesgueMeasure",
]
