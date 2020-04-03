from pbbo import *
import GPy
import evalset.test_funcs

# Select one of the following as the inference method: MCMCComparisonGP, EPComparisonGP, VIComparisonGPMF, VIComparisonGPFR
# We demonstrate here with the EP inference: 
inference = EPComparisonGP

# Select one of the following as the acquisition function: SumOfVariances, QExpectedImprovement, ThompsonSampling
# We demonstrate here with the Thompson sampling
acquisition = ThompsonSampling 

# Select objective, we demonstrate here with the 4 dimensional Sushi data
objective = evalset.test_funcs.Sushi()

# Normalize the objective and add noise:
objective.init_normalize_X() # scale bounds between 0 and 1
objective.init_normalize_Y() # scale function maximum and minimum between 0 and 1
noise_level = 0.05
objective = evalset.test_funcs.Noisifier(objective, 'add', noise_level) # add normally distributed noise with std 'noise_level'

# Generate a kernel for the GP
kernel = GPy.kern.RBF(input_dim=len(objective.bounds), ARD=True, variance=1.0, lengthscale=[0.5]*len(objective.bounds))
kernel.variance.constrain_fixed(1.0, warning=False)
kernel.lengthscale.constrain_fixed([0.5]*len(objective.bounds), warning=False)

# Some parameters for the acquisition function
options_acquisition = {    
    'acq_samples': 5000,                # How many samples are used in stochastic optimization of EV and QEI
    'acq_opt_restarts': 30,             # How many different initialization locations are randomly sampled for optimizing the acquisition function}
    'pool': -1,                         # How many posterior samples are used from the posterior, if -1 all are used. Only matters for MCMC.
}

# Acquisition function optimizer options, see possible options from scipy optimize options
optimizer_options = {}


# Options for the Bayesian optimization:
config = {
    'inference': inference,
    'kernel': kernel,
    'acquisition': acquisition(options_acquisition, optimizer_options),
    'batch_size': 4,                    # Batch size >2     
    'initial_size': 0,                  # If comparative feedback is used
    'initial_size_direct': 0,           # How many direct observations are used in initailization
    'noise': noise_level,               # Noise standard deviation for observations the likelihood uses
    'max_num_observations': 60,         # Max number of observation in the BO loo. Number of iterations is maximum number oservations divided by the batch size.
}

#Generate and run the Bayesian Optimization
bo = BayesianOptimization(config)
X, yc = bo.bayesian_optimization(objective) # X contains the points and yc contains the comparative observations


