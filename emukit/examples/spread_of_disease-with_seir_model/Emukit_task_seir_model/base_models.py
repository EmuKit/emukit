# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


class SIR(object):
    """
    3 compartment model for the spread of a disease
    susceptible -> infected -> recovered
    """

    def __init__(self, N: int, alpha: float):
        """
        :param N: the population size
        :param alpha: the ratio of infection rate and recovery rate
        """
        self.N = N
        self.alpha = alpha

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha


class SEIR(SIR):
    """
    4 compartment model for the spread of a disease
    susceptible -> exposed -> infected -> recovered
    """

    def __init__(self, N: int, alpha: float, beta: float):
        """
        :param N: the population size
        :param alpha: the ratio of infection rate and recovery rate
        :param beta: the ratio of incubation rate and recovery rate
        """
        super(SEIR, self).__init__(N=N, alpha=alpha)
        self.beta = beta
