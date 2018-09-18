# Emukit

Emukit is a highly adaptable Python toolkit for enriching decision making under uncertainty. This is particularly pertinent to complex systems where data is scarce or difficult to acquire. In these scenarios, propagating well-calibrated uncertainty estimates within a design loop or computational pipeline ensures that constrained resources are used effectively.

The main features currently available in Emukit are:

* **Multi-fidelity emulation:** build surrogate models when data is obtained from multiple information sources that have different fidelity and/or cost;
* **Bayesian optimisation:** optimise physical experiments and tune parameters of machine learning algorithms;
* **Experimental design/Active learning:** design the most informative experiments and perform active learning with machine learning models;
* **Sensitivity analysis:** analyse the influence of inputs on the outputs of a given system;
* **Bayesian quadrature [coming soon]:** efficiently compute the integrals of functions that are expensive to evaluate.

Emukit is agnostic to the underlying modelling framework, which means you can use any tool of your choice in the Python ecosystem to build the machine learning model, and still be able to use Emukit.

## Installation

Currently only installation from sources is supported.

### Dependencies / Prerequisites
Emukit's primary dependencies are Numpy, GPy and GPyOpt.
See [requirements](requirements/requirements.txt).

### Install from sources
To install Emukit from source, create a local folder where you would like to put Emukit source code, and run following commands:
```
git clone https://github.com/amzn/Emukit.git
cd Emukit
python setup.py install
```

Alternatively you can run
```
pip install git+https://github.com/amzn/Emukit.git
```

## Getting started
For examples see our [tutorial notebooks](examples/notebooks).

## API documentation
Coming soon!

## License

Emukit is licensed under Apache 2.0. Please refer to [LICENSE](LICENSE) and [NOTICE](NOTICE) for further license information.