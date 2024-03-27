# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from enum import Enum


class AcquisitionType(Enum):
    EI = 1
    PI = 2
    NLCB = 3


class ModelType(Enum):
    RandomForest = 1
    BayesianNeuralNetwork = 2
