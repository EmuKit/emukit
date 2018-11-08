---
layout: post
title:  "Bayesian Optimization"
categories: jekyll update
img: bayesian_optimization.jpeg
categories: two
---

Bayesian optimization is a sequential decision making approach to find the optimum of expensive to evaluate objective functions.


Many problems in machine learning, 



```python
from emukit.test_functions import branin_function
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.experimental_design.model_free.random_design import RandomDesign
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper

# Define objective function and parameter space
f, _ = branin_function()
parameter_space = ParameterSpace([ContinuousParameter('x1', -5, 10), 
                                  ContinuousParameter('x2', 0, 15)])


# Collect some point of the objective and define a model
design = RandomDesign(parameter_space)
X = design.get_samples(num_data_points = 5)
Y = f(X)

# Fit and wrap a model
model_gpy = GPRegression(X,Y)
model_emukit = GPyModelWrapper(model_gpy)
```


```python
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop 
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement 
from emukit.core.optimization import AcquisitionOptimizer 
from emukit.core.loop import FixedIterationsStoppingCondition 


# Load core elements for Bayesian optimization
expected_improvement = ExpectedImprovement(model = model_emukit)
optimizer            = AcquisitionOptimizer(space = parameter_space)
point_calculator     = Sequential(expected_improvement, optimizer)

# Create the Bayesian optimization object
bayesopt_loop = BayesianOptimizationLoop(model = model_emukit,
                                         space = parameter_space,
                                         acquisition = expected_improvement,
                                         batch_size = 5)

# Run the loop and extract the optimum
stopping_condition = FixedIterationsStoppingCondition(i_max = 10)
bayesopt_loop.run_loop(f, stopping_condition)
```

