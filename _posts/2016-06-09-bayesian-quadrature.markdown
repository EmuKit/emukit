---
layout: post
title:  "Bayesian Quadrature"
categories: jekyll update
img: bayesian_quadrature.png
---

Bayesian quadrature [[1, 2]](#refereces-on-quadrature) is an active learning method for the value of an integral given queries of the integrand 
on a finite and usually small amount of input locations.

Bayesian quadrature returns not only an estimator, but a full posterior distribution on the integral value
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

Thus, the success of Bayesian quadrature is based on three things: i) replacing an intractable integral with an inference 
problem on the integrand function ii) replacing the actual integration with an easier, analytic integration of the surrogate model
on the integrand function, and iii) actively choosing locations for integrand evaluations such that the budget is optimally used
in the sense encoded by the acquisition scheme.

#### Bayesian Quadrature in Emukit
Emukit provides basic functionality for vanilla Bayesian quadrature [[1, 2]](#refereces-on-quadrature), 
but also for more elaborate methods like WSABI [[3]](#refereces-on-quadrature). It alleviates the need to research and 
implement the integration of the emulator, the acquisition scheme and the active loop while providing a frame to implement 
and try out novel Bayesian quadrature methods.


We're always open to contributions! Please read our [contribution guidelines](CONTRIBUTING.md) for more information. 
We are particularly interested in contributions regarding translations and tutorials. 


#### Refereces on Quadrature

- [1] O'Hagan (1991) [Bayes-Hermite Quadrature](https://www.sciencedirect.com/science/article/pii/037837589190002V), *Journal of Statistical Planning and Inference* 29, pp. 245--260.
- [2] Diaconis (1988) [Bayesian numerical analysis](http://probabilistic-numerics.org/assets/pdf/Diaconis_1988.pdf), *Statistical decision theory and related topics* V, pp. 163--175.
- [3] Gunter et al. (2014) [Sampling for Inference in Probabilistic Models with Fast Bayesian Quadrature](https://papers.nips.cc/paper/5483-sampling-for-inference-in-probabilistic-models-with-fast-bayesian-quadrature), *Advances in Neural Information Processing Systems*, 27, pp. 2789--2797.

