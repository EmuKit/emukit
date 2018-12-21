---
layout: post
title:  "Bayesian Quadrature"
categories: jekyll update
img: bayesian_quadrature.png
---

Bayesian quadrature is an active learning method for the value of an integral given queries of the integrand 
on a finite and usually small amount of input locations.

Bayesian quadrature [[1, 2]](#refereces-on-quadrature) returns not only an estimator, but a full posterior distribution on the integral value
which can subsequently be used in decision making or uncertainty analysis.

Bayesian quadrature is especially useful when integrand evaluations are expensive and sampling schemes 
prohibitive, for example when evaluating the integrand involves running a complex computer simulation, a real-world experiment,
or a lab. But even when evaluation cost is manageable, the sheer amount of queries that might be required by classical 
algorithms is usually incentive enough to favor a smarter and less wasteful approach.

Instead, Bayesian quadrature draws information from structure encoded as prior over the function space modeling the integrand, 
such as regularity or smoothness, and is thus able to learn faster from fewer datapoints (function evaluations). 
In Emukit, the surrogate model for the integrand would be the *emulator* and the integrand that can be queried, and
 possibly describes a real, complex system, the *simulator*.

#### Bayesian Quadrature in the Loop

Like other sequential learning schemes, Bayesian quadrature iteratively selects points where the integrand will be queried 
next such that an acquisition function (e.g., the reduction of the integral variance) is maximized in each step. 
At the same time, the model class (often a Gaussian process) is updated with the newly collected datapoint and 
refined at each step by optimizing its hyperparameters. 
The actual integration that yields the distribution over the integral value is then performed by integrating the emulator
of the integrand function, which is often analytic for certain choices of Gaussian processes, or 
an analytic approximation (e.g., for WSABI [[3]](#refereces-on-quadrature)). 

Thus, the success of Bayesian quadrature is based on three things: i) replacing an intractable integral with a regression 
problem on the integrand function ii) replacing the actual integration with an easier, analytic integration of the surrogate model
on the integrand function, and iii) actively choosing locations for integrand evaluations such that the budget is optimally used
in the sense encoded by the acquisition scheme.

#### Bayesian Quadrature in Emukit
Emukit at the moment provides basic functionality for vanilla Bayesian quadrature where a Gaussian process surrogate model is placed upon 
the integrand which is then integrated directly. 
This is how it is done:

First we define the function that we want to integrate, also called the *integrand*. Here we choose the 1-dimensional
Hennig1D function (see [here](http://nbviewer.jupyter.org/github/amzn/emukit/blob/master/notebooks/Emukit-tutorial-Bayesian-quadrature-introduction.ipynb) 
for a visualization) which is already implemented in Emukit. We also choose the integration bounds: a lower bound and an upper bound.

```python
from emukit.test_functions import hennig1D

user_function = hennig1D()[0]
lb = -3. # lower integral bound
ub = 3. # upper integral bound
```

Next we choose three locations for some initial evaluations to get an initial model of the integrand, also called the initial design.
Here we use the GP regression model of [GPy](https://github.com/SheffieldML/GPy) since a wrapper already exists in Emukit. Note that in BQ we are usually restricted
in the choice of the kernel function. Emukit at the moment supports the RBF/Gaussian kernel for Bayesian quadrature.

```python
import GPy

X = np.array([[-2.],[-0.5], [-0.1]])
Y = user_function(X) # inital integrand evaluations at locations X 
gpy_model = GPy.models.GPRegression(X=X, Y=Y, 
                                    kernel=GPy.kern.RBF(input_dim=X.shape[1], 
                                    lengthscale=0.5, 
                                    variance=1.0))
```

Now we convert the [GPy](https://github.com/SheffieldML/GPy) GP model into an Emukit quadrature GP. 
Note that we also need to wrap the RBF kernel of the GPy model since Bayesian quadrature essentially integrates the kernel function. 

```python
from emukit.quadrature.kernels import QuadratureRBF
from emukit.model_wrappers.gpy_quadrature_wrappers import RBFGPy, \
BaseGaussianProcessGPy

emukit_rbf = RBFGPy(gpy_model.kern)
emukit_qrbf = QuadratureRBF(emukit_rbf, integral_bounds=[(lb, ub)])
emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)
```

There are several Bayesian quadrature methods out there. Emukit at the moment supports vanilla Bayesian quadrature where
the GP model is directly placed over the integrand function and then integrated analytically. Note that the integration method is different from the GP model,
for example other approaches (e.g., [[3]](#refereces-on-quadrature)) first transform the GP model before they integrate it.

```python
from emukit.quadrature.methods import VanillaBayesianQuadrature

emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model)
```

Now we define the active learning loop. The essential piece in the loop is the acquisition function. The vanilla BQ loop 
by default uses the integral-variance-reduction acquisition (IVR) which is a global quantity of the space.

```python
from emukit.quadrature.loop import VanillaBayesianQuadratureLoop

emukit_loop = VanillaBayesianQuadratureLoop(model=emukit_method)
```

Finally, we run the loop for `num_iter = 20` iterations. This will collect 20 additional observations, chosen by 
optimizing the acqusition function at every step. After each newly collected observations, the vanilla BQ model is updated 
and fitted to the new dataset.

```python                           
num_iter = 20          
emukit_loop.run_loop(user_function=hennig1D()[0], stopping_condition=num_iter)
```

And that's it! You can retrieve the integral and variance estimator by running
 
```python
integral_mean, integral_variance = emukit_loop.model.integrate()
``` 



Check our list of [notebooks](http://nbviewer.jupyter.org/github/amzn/emukit/blob/master/notebooks/index.ipynb) and [examples](https://github.com/amzn/emukit/tree/master/emukit/examples) if you want to learn more about how to do Bayesian quadrature and other methods with Emukit. You can also check the Emukit [documentation](https://emukit.readthedocs.io/en/latest/).

We’re always open to contributions! Please read our [contribution guidelines](https://github.com/amzn/emukit/blob/master/CONTRIBUTING.md) for more information. We are particularly interested in contributions
regarding examples and tutorials.

#### Refereces on Quadrature

- [1] O'Hagan (1991) [Bayes-Hermite Quadrature](https://www.sciencedirect.com/science/article/pii/037837589190002V), *Journal of Statistical Planning and Inference* 29, pp. 245--260.
- [2] Diaconis (1988) [Bayesian numerical analysis](http://probabilistic-numerics.org/assets/pdf/Diaconis_1988.pdf), *Statistical decision theory and related topics* V, pp. 163--175.
- [3] Gunter et al. (2014) [Sampling for Inference in Probabilistic Models with Fast Bayesian Quadrature](https://papers.nips.cc/paper/5483-sampling-for-inference-in-probabilistic-models-with-fast-bayesian-quadrature), *Advances in Neural Information Processing Systems*, 27, pp. 2789--2797.

