# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

class Forrester():
	'''Forrester function.'''
	def __init__(self,sd =np.float64(0)) -> None:
		'''
		:param sd: standard deviation of the outputs.
		'''
		self.input_dim = 1
		if sd==None: self.sd = 0
		else: self.sd=sd
		self.min = 0.78
		self.fmin = -6
		self.bounds = [(0,1)]

	def f(self,X: np.array) -> np.array:
		'''
		Forrester function

		:param X: input vector to be evaluated
		:return: outputs of the function
		'''
		X = X.reshape((len(X),1))
		n = X.shape[0]
		fval = ((6*X -2)**2)*np.sin(12*X-4)
		if self.sd ==0:
			noise = np.zeros(n).reshape(n,1)
		else:
			noise = np.random.normal(0,self.sd,n).reshape(n,1)
		return fval.reshape(n,1) + noise
