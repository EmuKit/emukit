import numpy as np

from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop

import GPy
import numpy as np
import math
import logging
import emukit
import evalset.test_funcs
from typing import Dict, Tuple
from functools import partial

from . import util
from . import ComparisonGP, ComparisonGPEmukitWrapper
from .acquisitions import AcquisitionFunction, EmukitAcquisitionFunctionWrapper, ThompsonSampling, SequentialGradientAcquisitionOptimizer 


def create_bayesian_optimization_loop(gpy_model: ComparisonGP, lims: np.array, batch_size: int,
                                      acquisition: AcquisitionFunction) -> BayesianOptimizationLoop:
    """
    Creates Bayesian optimization loop for Bayesian neural network or random forest models.
    :param gpy_model: the GPy model used in optimization
    :param lims: Optimization limits for the inputs
    :param batch_size: number of observations used in batch
    :param acquisition: acquisition function used in the bayesian optimization
    :return: emukit BO loop
    """

    # Create model
    model = ComparisonGPEmukitWrapper(gpy_model, batch_size)
    
    # Create acquisition
    emukit_acquisition = EmukitAcquisitionFunctionWrapper(model, acquisition)

    if type(emukit_acquisition.acquisitionFunction) is ThompsonSampling:
        parameter_space = []
        for j in range(len(lims)):
            parameter_space += [ContinuousParameter('x{}'.format(j), lims[j][0], lims[j][1])]
        parameter_space = ParameterSpace(parameter_space)
        acquisition_optimizer = SequentialGradientAcquisitionOptimizer(parameter_space, batch_size)
    else:
        parameter_space = []
        for k in range(batch_size):
            for j in range(len(lims)):
                parameter_space += [ContinuousParameter('x{}{}'.format(k,j), lims[j][0], lims[j][1])]
        parameter_space = ParameterSpace(parameter_space)
        acquisition_optimizer = GradientAcquisitionOptimizer(parameter_space)
    
    bo = BayesianOptimizationLoop(model=model, space=parameter_space, acquisition=emukit_acquisition)
    return BayesianOptimizationLoop(model=model, space=parameter_space, acquisition=emukit_acquisition, acquisition_optimizer=acquisition_optimizer)


class BayesianOptimization():
    """
    This is a simple implementation of Bayesian optimization needed for preforming preferential batch Bayesian optimization
       
    The class structure is heavily inspired by https://github.com/oxfordcontrol/Bayesian-Optimization/blob/GPy-based/methods/bo.py
    
    :param options: a dictionary containing otions needed by the BO loop.
                    The dictionary must contain values for 'objective' (type: TestFunction),
                    'inference' (type: CompartisonGP), 'acquisition' (type: AcquisitionFunction),
                    'kernel' (type: GPy.kern.Kern) and 'batch_size' (type:int)
    """
    def __init__(self, options: Dict):        
        self.options = options.copy()
        
        self.kernel = options['kernel'].copy()                          # GPy kernel used for the optimization 
        self.inference = options['inference']                           # Inference method used to approximate the posterior
        self.acquisition = options['acquisition']                       # The acquisition function used to get new points
        self.batch_size = options['batch_size']                         # Number of observations in a batch
        self.max_num_observations = options['max_num_observations']     # Maximum number of observations allowed
        
        # Number of iterations the BO loop is ran can be computed from the max number of observations and batch size
        self.iterations = math.ceil(float(self.max_num_observations)/float(self.batch_size))
        
        # Optional parameters that have defaults 
        self.noise = options.get('noise', 1e-6)                         # Noise std the likelihood uses for observations
        
        # If we use comparative observations from the objective function before starting
        self.use_comparative_observations_in_init = options.get('use_comparative_observations_in_init', False)
        # If we use direct observations from the objective function before starting
        self.use_direct_observations_in_init = options.get('use_direct_observations_in_init', False)
        # Boolean indicating if the locations for the initial draws are chosen at random or selected uniformly from a grid
        self.random = options.get('random', False)

        self.log_file = options.get('log_file', None)                   # Log file used to store all print outputs during the BO
        
        # Configure logger
        util.configure_logger(log_file=self.log_file)

    def get_logger(self) -> logging.Logger:
        """
        Get the logging function used to save the print outputs
        :return: a logger where the prints are directed
        """
        return logging.getLogger(self.log_file)
    
    def bayesian_optimization(self, objective: evalset.test_funcs.TestFunction) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function implements the main loop of Bayesian Optimization,
        
        The function takes the objective 
        
        The function returns the final set of points X_all, the respective
        batch feedbacks yc_all
        
        :param objective: The black box function to be optimized
        :return: The locations the function has been evaluated so far and the comparison outcomes
        """
    
        # SET UP THE GP MODEL #
        bounds = objective.bounds
        dim = len(bounds)
        
        lik = GPy.likelihoods.Gaussian()
        lik.variance.constrain_fixed(self.noise**2, warning=False)
        noise = self.noise

        X0 = np.empty((0,dim))
        y = []
        yc = []
        
        def objective_modifier(x, f=None, batch_size=1):
            return np.concatenate(tuple( f(x[:,i*batch_size:(i+1)*batch_size]).reshape((-1,1)) for i in range(x.shape[1]//batch_size)), axis=1)
                
        
        # Initial observations:
        if self.use_comparative_observations_in_init:
            if self.random:
                X0 = util.random_sample(bounds, 2**dim)
            else:
                X0 = util.grid_sample(dim)
            yc = util.give_comparisons(objective.f, X0)
        if self.use_direct_observations_in_init:
            if self.random:
                Xn = util.random_sample(bounds, 2**dim)
            else:
                Xn = util.grid_sample(dim)
            yn = objective.f(Xn).reshape((-1,1))
            y = [(X0.shape[0] + i, yi) for i,yi in enumerate(yn)]
            X0 = np.concatenate((X0, Xn), axis=0)
            
            
        if not self.use_comparative_observations_in_init and not self.use_direct_observations_in_init:
            m = self.inference(util.static_sample(bounds), [(i, yi) for i,yi in enumerate(np.array([[0], [0]]))], yc, self.kernel.copy(), lik, get_logger=self.get_logger)
        else:
            m = self.inference(X0, y, yc, self.kernel.copy(), lik, get_logger=self.get_logger)

        # CREATE BO LOOP #
        bo_loop = create_bayesian_optimization_loop(m, bounds, self.batch_size, self.acquisition)
        
        # RUN THE LOOP #
        bo_loop.run_loop( partial(objective_modifier, f=objective.f, batch_size=self.batch_size), self.iterations)
        return m.X, m.yc
