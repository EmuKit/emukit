# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

class Ishigami(object):
    '''
    Ishigami function was first introduced in 'Ishigami and T. Homma, An importance quantification
    technique in uncertainty analysis for computer 556 models, in Uncertainty Modeling and Analysis,
    1990. Proceedings., First International Symposium 557 on, IEEE, 1990, pp. 398–403'. This class
    contains: the elements of the Sobol decomposition of the Ishigami fucntion, three low fidelity
    approximations, and the values of the main effects of the Sobol components:

    .. math::

    f(x_1,x_2,x_3) = \sum f_i(x_i) + \sum f_{ij}(x_i,x_j) + f(x_1,x_2,x_3)

    '''

    def __init__(self, a, b):
        self.a = a
        self.b = b

        self.variance_total = 0.5 + self.a**2/8 + np.pi**4*self.b/5 + np.pi**8*self.b**2/18
        self.variance_x1 = 0.5*(1+self.b*np.pi**4/5)**2
        self.variance_x2 = self.a**2/8
        self.variance_x3 = 0
        self.variance_x12 = 0
        self.variance_x13 = np.pi**8*self.b**2*(1/18-1/50)
        self.variance_x23 = 0
        self.variance_x123 = 0

        ### Main effects
        self.main_effects = {}
        self.main_effects['x1'] = self.variance_x1/self.variance_total
        self.main_effects['x2'] = self.variance_x2/self.variance_total
        self.main_effects['x3'] = self.variance_x3/self.variance_total

        ### Total effects
        self.total_effects = {}
        self.total_effects['x1'] = (self.variance_x1+self.variance_x13)/self.variance_total
        self.total_effects['x2'] = (self.variance_x2)/self.variance_total
        self.total_effects['x3'] = (self.variance_x3+self.variance_x13)/self.variance_total

    ### --- 4 fidelities (#1 is the highest fidelity)
    def fidelity1(self,x: np.ndarray) -> np.ndarray:
        return np.sin(x[:,0]) + self.a*np.sin(x[:,1])**2 + self.b*x[:,2]**4*(np.sin(x[:,0]))

    def fidelity2(self,x: np.ndarray) -> np.ndarray:
        return np.sin(x[:,0]) + self.a*np.sin(x[:,1]) + self.b*x[:,2]**4*(np.sin(x[:,0]))

    def fidelity3(self,x: np.ndarray) -> np.ndarray:
        return np.sin(x[:,0]) + .95*self.a*np.sin(x[:,1])**2 + self.b*x[:,2]**4*(np.sin(x[:,0]))

    def fidelity4(self,x: np.ndarray) -> np.ndarray:
        return np.sin(x[:,0]) + .6*self.a*np.sin(x[:,1])**2 + 9*self.b*x[:,2]**4*(np.sin(x[:,0]))

    ### --- Main effects
    def f0(self):
        return self.a/2

    def f1(self,x1):
        return (1+ (self.b/5)*np.pi**4)*np.sin(x1)

    def f2(self,x2):
        return self.a *np.sin(x2)**2-0.5*self.a

    def f3(self,x3):
        return x3[:,0]*0

    def f12(self,x12):
        return x12[:,0]*0

    def f13(self,x13):
        return self.b * np.sin(x13[:,0])*(x13[:,1]**4 - np.pi**4/5)

    def f23(self,x23):
        return x23[:,0]*0

    def f123(self,x123):
        return x123[:,0]*0
