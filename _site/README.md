# Emukit

[![Build Status](https://travis-ci.org/amzn/emukit.svg?branch=master)](https://travis-ci.org/amzn/emukit) |
[![Tests Coverage](https://codecov.io/gh/amzn/emukit/branch/master/graph/badge.svg)](https://codecov.io/gh/amzn/emukit) |
[![GitHub License](https://img.shields.io/github/license/amzn/emukit.svg)](https://github.com/amzn/emukit/blob/master/LICENSE)

Emukit is a highly adaptable Python toolkit for enriching decision making under uncertainty. This is particularly pertinent to complex systems where data is scarce or difficult to acquire. In these scenarios, propagating well-calibrated uncertainty estimates within a design loop or computational pipeline ensures that constrained resources are used effectively.

The main features currently available in Emukit are:

* **Multi-fidelity emulation:** build surrogate models when data is obtained from multiple information sources that have different fidelity and/or cost;
* **Bayesian optimisation:** optimise physical experiments and tune parameters of machine learning algorithms;
* **Experimental design/Active learning:** design the most informative experiments and perform active learning with machine learning models;
* **Sensitivity analysis:** analyse the influence of inputs on the outputs of a given system;
* **Bayesian quadrature [coming soon]:** efficiently compute the integrals of functions that are expensive to evaluate.

Emukit is agnostic to the underlying modelling framework, which means you can use any tool of your choice in the Python ecosystem to build the machine learning model, and still be able to use Emukit.

## Installation
* Fork the repository
* Go to settings and set Github Pages source as master.
* Your new site should be ready.

For more themes visit - [https://jekyll-themes.com](https://jekyll-themes.com)