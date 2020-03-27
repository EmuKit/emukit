"""
NOTE: This file is modified from https://github.com/sigopt/evalset/blob/master/evalset/test_funcs.py
We have added only the "DataFunction" class and functions to evaluate the real functions
below is the copyright notice for the file:

This file offers access to functions used during the development of the article
    A Stratified Analysis of Bayesian Optimization Methods

It incorporates functions developed/collected for the AMPGO benchmark by Andrea Gavana <andrea.gavana@gmail.com>
As of January 2016, the website http://infinity77.net/global_optimization/test_functions.html hosts images
of some of these functions.

Each function has an evaluate, which accepts in a single axis numpy.array x and returns a single value.
None of these functions allows for vectorized computation.

NOTE: These functions are designed to be minimized ... the paper referenced above talks about maximization.
    This was intentional to fit the standard of most optimization algorithms.

Some of these functions work in different dimensions, and some have a specified dimension.  The assert statement
will prevent incorrect calls.

For some of these functions, the maximum and minimum are determined analytically.  For others, there is only
a numerical determination of the optimum.  Also, some of these functions have the same minimum value at multiple
locations; if that is the case, only the location of one is provided.

Each function is also tagged with a list of relevant classifiers:
  boring - A mostly boring function that only has a small region of action.
  oscillatory - A function with a general trend and an short range oscillatory component.
  discrete - A function which can only take discrete values.
  unimodal - A function with a single local minimum, or no local minimum and only a minimum on the boundary.

multimodal - A function with multiple local minimum
  bound_min - A function with its minimum on the boundary.
  multi_min - A function which takes its minimum value at multiple locations.
  nonsmooth - A function with discontinuous derivatives.
  noisy - A function with a base behavior which is clouded by noise.
  unscaled - A function with max value on a grossly different scale than the average or min value.
  complicated - These are functions that may fit a behavior, but not in the most obvious or satisfying way.

The complicated tag is used to alert users that the function may have interesting or complicated behavior.
As an example, the Ackley function is technically oscillatory, but with such a short wavelength that its
behavior probably seems more like noise.  Additionally, it is meant to act as unimodal, but is definitely
not, and it may appear mostly boring in higher dimensions.

The MIT License (MIT)

Copyright (c) 2016 SigOpt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy

from numpy import abs, arange, arctan2, asarray, cos, exp, floor, log, log10, mean
from numpy import pi, prod, roll, seterr, sign, sin, sqrt, sum, zeros, zeros_like, tan
from numpy import dot, inner

from scipy.special import jv as besselj
from scipy.interpolate import LinearNDInterpolator
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from typing import List


def lzip(*args):
    """
    Zip, but returns zipped result as a list.
    """
    return list(zip(*args))

class BlackBoxFunction:
    '''
    All functions to be optimized with BayesianOptimization class
    or DerivativeBayesianOptimization class must inherit this class
    '''

    def __init__(self):
        pass

    def get_dim(self):
        '''
        Should return the size of the input space
        '''
        raise NotImplementedError

    def do_evaluate(self, x):
        '''
        returns the possibly stochastic evaluation for given x
        '''
        raise NotImplementedError

    def do_evaluate_clean(self, x):
        '''
        If possible, returns the noiseless evaluation for given x
        '''
        return None

    def f(self, x):
        raise NotImplementedError


class TestFunction(BlackBoxFunction):
    """
    The base class from which functions of interest inherit.
    """

    __metaclass__ = ABCMeta


    def __init__(self, dim, verify=True):
        assert dim > 0
        self.dim = dim
        self.verify = verify
        self.num_evals = 0
        self.min_loc = None
        self.fmin = None
        self.local_fmin = []
        self.fmax = None
        self.bounds = None
        self.classifiers = []

        self.records = None
        self.noise_std = 0.0
        self.lengths=None

        self.deviation = 1.0
        bounds_array, lengths = self.tuplebounds_2_arrays(lzip([0] * self.dim, [1] * self.dim))
        self.us_bounds = bounds_array
        self.lengths = lengths
        self.reset_records()

    def init_normalize_Y(self):
        self.deviation = self.fmax-self.fmin

    def init_normalize_X(self):
        bounds_array, lengths = self.tuplebounds_2_arrays(self.bounds)
        self.lengths = lengths
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.us_bounds = bounds_array
        self.min_loc_01 = (self.min_loc-self.us_bounds[:,0])/self.lengths

    def tuplebounds_2_arrays(self, bounds):
        bounds_array = numpy.zeros((self.dim,2))
        lengths = numpy.zeros((self.dim))
        for i in range(self.dim):
            bounds_array[i,0] = bounds[i][0]
            bounds_array[i,1] = bounds[i][1]
            lengths[i] = bounds[i][1]- bounds[i][0]
        return bounds_array, lengths

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self.dim)

    def get_dim(self):
        return self.dim

    def evaluate(self, x):
        if self.verify and (not isinstance(x, numpy.ndarray) or x.shape != (self.dim,)):
            raise ValueError('Argument must be a numpy array of length {}'.format(self.dim))

        self.num_evals += 1
        value = self.do_evaluate(x)
        to_be_returned = value.item() if hasattr(value, 'item') else value
        self.update_records(now(), x, to_be_returned)
        # Convert numpy types to Python types
        return to_be_returned

    def update_records(self, time, location, value):
        self.records['time'].append(time)
        self.records['locations'].append(location)
        self.records['values'].append(value)

    def reset_records(self):
        self.records = {'time': [], 'locations': [], 'values': []}

    def f_c(self, x):
        '''
        returns function values when x is given in  numpy array of shape N x d where N is number of points and d is dimension
        '''
        x = numpy.atleast_2d(x)
        y = numpy.array([(self.do_evaluate_clean( (x[i,:]*self.lengths + self.us_bounds[:,0]) ) - self.fmin)/self.deviation - 1.0  for i in range(x.shape[0])])
        return y.flatten()

    def f(self, x):
        '''
        returns function values when x is given in  numpy array of shape N x d where N is number of points and d is dimension
        '''

        x = numpy.atleast_2d(x)
        y = numpy.array([(self.do_evaluate_clean( (x[i,:]*self.lengths + self.us_bounds[:,0]) ) - self.fmin)/self.deviation - 1.0 \
            + numpy.random.normal(0, self.noise_std,1) for i in range(x.shape[0])])
        return y.flatten()

    @abstractmethod
    def do_evaluate(self, x):
        """
        :param x: point at which to evaluate the function
        :type x: numpy.array with shape (self.dim, )
        """
        raise NotImplementedError

    @abstractmethod
    def do_evaluate_clean(self, x):
        """
        :param x: point at which to evaluate the function
        :type x: numpy.array with shape (self.dim, )
        """
        return self.do_evaluate(x)


class Noisifier(TestFunction):
    """
    This class dirties function evaluations with Gaussian noise.

    If type == 'add', then the noise is additive; for type == 'mult' the noise is multiplicative.
    sd defines the magnitude of the noise, i.e., the standard deviation of the Gaussian.

    Example: ackley_noise_addp01 = Noisifier(Ackley(3), 'add', .01)

    Obviously, with the presence of noise, the max and min may no longer be accurate.
    """
    def __init__(self, func, noise_type, level, verify=True):
        assert isinstance(func, TestFunction)
        if level < 0:
            raise ValueError('Noise level must be positive, level={0}'.format(level))
        super(Noisifier, self).__init__(func.dim, verify)
        self.bounds, self.min_loc, self.fmax, self.fmin = func.bounds, func.min_loc, func.fmax, func.fmin
        self.type = noise_type
        self.level = level
        self.func = func
        self.func_clean = func
        self.classifiers = list(set(self.classifiers) | set(['noisy']))

    def do_evaluate(self, x):
        if self.type == 'add':
            return self.func.do_evaluate(x) + self.level * numpy.random.normal()
        else:
            return self.func.do_evaluate(x) * (1 + self.level * numpy.random.normal())

    def __repr__(self):
        return '{0}({1!r}, {2}, {3})'.format(
            self.__class__.__name__,
            self.func,
            self.type,
            self.level,
        )

    def do_evaluate_clean(self, x):
        return self.func.do_evaluate_clean(x)

class Adjiman(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Adjiman, self).__init__(dim)
        self.bounds = ([-1, 2], [-1, 1])
        self.min_loc = [2, 0.10578]
        self.fmin = -2.02180678
        self.fmax = 1.07715029333
        self.classifiers = ['unimodal', 'bound_min']

    def do_evaluate(self, x):
        x1, x2 = x
        return cos(x1) * sin(x2) - x1 / (x2 ** 2 + 1)

class Deceptive(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Deceptive, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.min_loc = [.333333, .6666666]
        self.fmin = -1
        self.fmax = 0
        self.classifiers = ['nonsmooth']

    def do_evaluate(self, x):
        alpha = asarray(self.min_loc)
        beta = 2
        g = zeros((self.dim, ))
        for i in range(self.dim):
            if x[i] <= 0:
                g[i] = x[i]
            elif x[i] < 0.8 * alpha[i]:
                g[i] = -x[i] / alpha[i] + 0.8
            elif x[i] < alpha[i]:
                g[i] = 5 * x[i] / alpha[i] - 4
            elif x[i] < (1 + 4 * alpha[i]) / 5:
                g[i] = 5 * (x[i] - alpha[i]) / (alpha[i] - 1) + 1
            elif x[i] <= 1:
                g[i] = (x[i] - 1) / (1 - alpha[i]) + .8
            else:
                g[i] = x[i] - 1
        return -((1.0 / self.dim) * sum(g)) ** beta

class Hartmann3(TestFunction):
    def __init__(self, dim=3):
        assert dim == 3
        super(Hartmann3, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.min_loc = [0.1, 0.55592003, 0.85218259]
        self.fmin = -3.86278214782076
        self.fmax = -3.77271851416e-05

    def do_evaluate(self, x):
        a = asarray([[3,  0.1,  3,  0.1],
                     [10, 10, 10, 10],
                     [30, 35, 30, 35]])
        p = asarray([[0.36890, 0.46990, 0.10910, 0.03815],
                     [0.11700, 0.43870, 0.87320, 0.57430],
                     [0.26730, 0.74700, 0.55470, 0.88280]])
        c = asarray([1, 1.2, 3, 3.2])
        d = zeros_like(c)
        for i in range(4):
            d[i] = sum(a[:, i] * (x - p[:, i]) ** 2)
        return -sum(c * exp(-d))


class Hartmann4(TestFunction):
    def __init__(self, dim=4):
        assert dim == 4
        super(Hartmann4, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [1] * self.dim)
        self.min_loc = [0.49204492762, 0.82366439640, 0.30064257056, 0.55643899079]
        self.fmin = -3.93518472715
        self.fmax = 1.31104361811

    def do_evaluate(self, x):
        a = asarray([[10, 3, 17, 3.5, 1.7, 8],
                     [.05, 10, 17, .1, 8, 14],
                     [3, 3.5, 1.7, 10, 17, 8],
                     [17, 8, .05, 10, .1, 14]])
        p = asarray([[.1312, .1696, .5569, .0124, .8283, .5886],
                     [.2329, .4135, .8307, .3736, .1004, .9991],
                     [.2348, .1451, .3522, .2883, .3047, .6650],
                     [.4047, .8828, .8732, .5743, .1091, .0381]])
        c = asarray([1, 1.2, 3, 3.2])
        d = zeros_like(c)
        for i in range(4):
            d[i] = sum(a[:, i] * (x - p[:, i]) ** 2)
        return (1.1 - sum(c * exp(-d))) / 0.839

class HolderTable(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(HolderTable, self).__init__(dim)
        self.bounds = lzip([-10] * self.dim, [10] * self.dim)
        self.min_loc = [8.055023472141116, 9.664590028909654]
        self.fglob = -19.20850256788675
        self.fmin = -19.20850256788675
        self.fmax = 0
        self.classifiers = ['multi_min', 'bound_min', 'oscillatory', 'complicated']

    def do_evaluate(self, x):
        x1, x2 = x
        return -abs(sin(x1) * cos(x2) * exp(abs(1 - sqrt(x1 ** 2 + x2 ** 2) / pi)))


class Hosaki(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(Hosaki, self).__init__(dim)
        self.bounds = lzip([0] * self.dim, [5] * self.dim)
        self.min_loc = [4, 2]
        self.fmin = -2.3458
        self.fmax = 0.54134113295

    def do_evaluate(self, x):
        x1, x2 = x
        return (1 + x1 * (-8 + x1 * (7 + x1 * (-2.33333 + x1 * .25)))) * x2 * x2 * exp(-x2)

class McCourtBase(TestFunction):
    """
    This is a class of functions that all fit into the framework of a linear combination of functions, many of
    which are positive definite kernels, but not all.

    These were created by playing around with parameter choices for long enough until a function with desired
    properties was produced.
    """
    @staticmethod
    def dist_sq(x, centers, e_mat, dist_type=2):
        if dist_type == 1:
            ret_val = numpy.array([
                [numpy.sum(numpy.abs((xpt - center) * evec)) for evec, center in lzip(numpy.sqrt(e_mat), centers)]
                for xpt in x
            ])
        elif dist_type == 2:
            ret_val = numpy.array([
                [numpy.dot((xpt - center) * evec, (xpt - center)) for evec, center in lzip(e_mat, centers)]
                for xpt in x
            ])
        elif dist_type == 'inf':
            ret_val = numpy.array([
                [numpy.max(numpy.abs((xpt - center) * evec)) for evec, center in lzip(numpy.sqrt(e_mat), centers)]
                for xpt in x
            ])
        else:
            raise ValueError('Unrecognized distance type {0}'.format(dist_type))
        return ret_val

    def __init__(self, dim, kernel, e_mat, coefs, centers):
        super(McCourtBase, self).__init__(dim)
        assert e_mat.shape == centers.shape
        assert e_mat.shape[0] == coefs.shape[0]
        assert e_mat.shape[1] == dim
        self.kernel = kernel
        self.e_mat = e_mat
        self.coefs = coefs
        self.centers = centers
        self.bounds = [(0, 1)] * dim

    def do_evaluate(self, x):
        return_1d = False
        if len(x.shape) == 1:  # In case passed as a single vector instead of 2D array
            x = x[numpy.newaxis, :]
            return_1d = True
        assert self.e_mat.shape[1] == x.shape[1]  # Correct dimension
        ans = numpy.sum(self.coefs * self.kernel(x), axis=1)
        return ans[0] if return_1d else ans


class MixtureOfGaussians02(TestFunction):

    def __init__(self, dim=2):
        assert dim == 2
        super(MixtureOfGaussians02, self).__init__(dim)
        self.bounds = lzip([-1] * self.dim, [1] * self.dim)
        self.min_loc = [(-0.19945435737, -0.49900294852)]
        self.fmin = -0.70126732387
        self.fmax = -0.00001198419
        self.local_fmin = [-0.70126732387, -0.30000266214]
        self.classifiers = ['multimodal']

    def do_evaluate(self, x):
        x1, x2 = x
        return -(
            .7 * numpy.exp(-10 * (.8 * (x1 + .2) ** 2 + .7 * (x2 + .5) ** 2)) +
            .3 * numpy.exp(-8 * (.3 * (x1 - .8) ** 2 + .6 * (x2 - .3) ** 2))
        )

class SixHumpCamel(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(SixHumpCamel, self).__init__(dim)
        self.bounds = [[-2, 2], [-1.5, 1.5]]
        self.min_loc = [0.08984201368301331, -0.7126564032704135]
        self.fmin = -1.031628
        self.fmax = 17.98333333333
        self.classifiers = ['multi_min']

    def do_evaluate(self, x):
        x1, x2 = x
        return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (4 * x2 ** 2 - 4) * x2 ** 2


class UrsemWaves(TestFunction):
    def __init__(self, dim=2):
        assert dim == 2
        super(UrsemWaves, self).__init__(dim)
        self.bounds = [(-0.9, 1.2), (-1.2, 1.2)]
        self.min_loc = [1.2] * self.dim
        self.fmin = -8.5536
        self.fmax = 7.71938723147
        self.classifiers = ['bound_min']

    def do_evaluate(self, x):
        x1, x2 = x
        return (
            -0.9 * x1 ** 2 + (x2 ** 2 - 4.5 * x2 ** 2) * x1 * x2 +
            4.7 * cos(3 * x1 - x2 ** 2 * (2 + x1)) * sin(2.5 * pi * x1)
        )

# Below are all 1D functions
class Problem02(TestFunction):
    def __init__(self, dim=1):
        assert dim == 1
        super(Problem02, self).__init__(dim)
        self.bounds = [(2.7, 7.5)]
        self.min_loc = 5.145735285687302
        self.fmin = -1.899599349152113
        self.fmax = 0.888314780101

    def do_evaluate(self, x):
        x = x[0]
        return sin(x) + sin(3.33333 * x)



## The rest of the file is new


class DataFunction(TestFunction):
    """
    A base class for using data as a function.
    The class takes care of setting up the bounds,
    finding the locations of minimum and maximum 
    and interpolating in the points in between the
    data points
    
    Given the data (X) and observations (Y), forms a function that
    interpolates the values between the provided locations
    
    :param X: Input locations
    :param Y: Input observations
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        dim = X.shape[1]
        super(DataFunction, self).__init__(dim)

        self.X = X
        self.Y = Y

        self.bounds = lzip(numpy.min(X, axis=0), numpy.max(X, axis=0))
        self.min_loc = self.X[numpy.argmin(self.Y), :]

            
        #Append corners with score 0
        corners = DataFunction.give_corners(self.bounds)

        self.X = numpy.r_[self.X, corners]
        self.Y = numpy.r_[self.Y, numpy.max(self.Y)*numpy.ones((corners.shape[0],1))]
        self.interpolator = LinearNDInterpolator(self.X, self.Y, fill_value=numpy.nan)

        self.fmin = numpy.min(self.Y)
        self.fmax = numpy.max(self.Y)
        self.classifiers = ['complicated', 'oscillatory', 'unimodal', 'noisy']

    @staticmethod
    def give_corners(bounds: np.ndarray) -> np.ndarray:
        """
        Given the bounds, returns the corners of the hyperrectangle the data just barely fits
        
        :param bounds: Bounds of the data set (minimum and maximum in every dimension)
        :return: All corners of the dataset 
        """
        if len(bounds) > 1:
            corners = DataFunction.give_corners(bounds[1:])
            firsts = numpy.c_[numpy.ones((corners.shape[0],1))*bounds[0][0], corners]
            seconds = numpy.c_[numpy.ones((corners.shape[0],1))*bounds[0][1], corners]
            return numpy.r_[firsts, seconds]
        else:
            return numpy.array(bounds[-1]).reshape((-1,1))

    def do_evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the data function at the given location by interpolating
        
        :param x: location where the data is wanted to be interpolated
        :return: interpolation result
        """
        return self.interpolator(x)

class Sushi(DataFunction):
    """
    Sushi data as a function : http://www.kamishima.net/sushi/ 
    
    :params use_discrete_features: if True, also discrete features of data are used.
    :param num_features: how many of the continuous features are used.
    """
    def __init__(self, use_discrete_features: bool=False, num_features: int=4):
        # The data files:
        dirname, _ = os.path.split(os.path.abspath(__file__))
        df_features = pd.read_csv(os.path.realpath(os.path.join(dirname,"data-files", "sushi3.idata")), sep="\t", header=None)
        df_scores = pd.read_csv(os.path.realpath(os.path.join(dirname,"data-files", "sushi3b.5000.10.score")), sep=" ", header=None)
        # Select the features we want (in the order of importance defined by RBF ARD kernel)
        features = df_features.values[:, [6,5,7,8][:num_features] ]
        if use_discrete_features:
            discrete_features = df_features.values[:, [2, 3, 4]]
            features = numpy.concatenate((features, discrete_features), axis=1)

        # Generate scrores from preferences
        scores = []
        for item_a in range(df_scores.values.shape[1]):
            score = 0
            for item_b in range(df_scores.values.shape[1]):
                if Sushi._prefer_a(item_a, item_b, df_scores):
                    score += 1
            scores.append(score / float(df_scores.values.shape[1]))
        
        X = features
        Y = - numpy.array(scores).reshape((-1,1)) # Invert, since we want to find the maximum by BO which finds the minimum
        
        # Scale the data between 0 and 1
        X = (X - numpy.min(X, axis=0)[None, :])/(numpy.max(X, axis=0) - numpy.min(X, axis=0))[None, :]
        Y = (Y-numpy.min(Y))/(numpy.max(Y)-numpy.min(Y))
        
        super(Sushi, self).__init__(X, Y)

    @classmethod
    def _prefer_a(cls, item_a: int, item_b: int, df_scores: List):
        """
        Check from data if item_a has higher score that item_b
        
        :param item_a: index of the first item to be compared
        :param item_b: index of the second item to be compared
        :param df_scores: Scores of all dat points
        :return: True if item_a is preferred over item_b
        """
        ix = (df_scores[item_a].values > -1) * (df_scores[item_b].values > -1)
        prefer_a = numpy.mean(df_scores[item_a].values[ix] > df_scores[item_b].values[ix])
        prefer_b = numpy.mean(df_scores[item_b].values[ix] > df_scores[item_a].values[ix])
        return prefer_a > prefer_b

class Concrete(DataFunction):
    '''
    Concrete compressive strength data as a function: https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength 
    
    :param num_features: how many of the features are used.
    '''
    def __init__(self, num_features: int=3):
        data_file = "data.csv"
        importance = [2, 0, 3, 7, 1, 4, 5, 6]
        
        dirname, _ = os.path.split(os.path.abspath(__file__))
        data = pd.read_csv(os.path.realpath(os.path.join(dirname,"data-files", data_file)), sep=",", header=None)                

        X = data.values[:,0:-1]
        Y = -data.values[:,-1].reshape((-1,1))
        X = (X - numpy.min(X, axis=0)[None, :])/(numpy.max(X, axis=0) - numpy.min(X, axis=0))[None, :]
        Y = (Y-numpy.min(Y))/(numpy.max(Y)-numpy.min(Y))

        X = X[:, importance[:num_features]]
        super(Concrete, self).__init__(X, Y)

class Candy(DataFunction):
    '''
    Halloween candy data as a function: https://fivethirtyeight.com/features/the-ultimate-halloween-candy-power-ranking/ 
    '''
    def __init__(self):
        data_file = "candy-data.csv"
        dirname, _ = os.path.split(os.path.abspath(__file__))
        X = pd.read_csv(os.path.realpath(os.path.join(dirname,"data-files", data_file)), sep=",", header=None)
        
        Y = -X.values[:,-1].reshape((-1,1))        
        X = X.values[:,-3:-1]
        
        X = (X - numpy.min(X, axis=0)[None, :])/(numpy.max(X, axis=0) - numpy.min(X, axis=0))[None, :]
        Y = (Y-numpy.min(Y))/(numpy.max(Y)-numpy.min(Y))
        super(Candy, self).__init__(X, Y)

class Wine(DataFunction):
    '''
    White wine quality data as a function: https://archive.ics.uci.edu/ml/datasets/Wine+Quality 
    
    :param num_features: how many of the features are used.
    '''
    def __init__(self, num_features: int=4):
        data_file = "wine_data.csv"
        importance = [ 5, 3, 7, 6, 10, 4, 1, 8, 2, 0, 9] 

        dirname, _ = os.path.split(os.path.abspath(__file__))
        data = pd.read_csv(os.path.realpath(os.path.join(dirname,"data-files", data_file)), sep=",", header=None)
        X = data.values[:,0:-1]
        Y = -data.values[:,-1].reshape((-1,1))+1.0
        X = X[:, importance[:num_features]]
        super(Wine, self).__init__(X, Y)        