# Emukit

[![Build Status](https://github.com/EmuKit/emukit/workflows/Tests/badge.svg)](https://github.com/EmuKit/emukit/actions?query=workflow%3ATests) |
[![Documentation Status](https://readthedocs.org/projects/emukit/badge/?version=latest)](https://emukit.readthedocs.io/en/latest/?badge=latest) |
[![Tests Coverage](https://codecov.io/gh/emukit/emukit/branch/main/graph/badge.svg)](https://codecov.io/gh/emukit/emukit) |
[![GitHub License](https://img.shields.io/github/license/emukit/emukit.svg)](https://github.com/emukit/emukit/blob/main/LICENSE)

[Website](https://emukit.github.io/) |
[Documentation](https://emukit.readthedocs.io/) |
[Contribution Guide](CONTRIBUTING.md)

Emukit is a highly adaptable Python toolkit for enriching decision making under uncertainty. This is particularly pertinent to complex systems where data is scarce or difficult to acquire. In these scenarios, propagating well-calibrated uncertainty estimates within a design loop or computational pipeline ensures that constrained resources are used effectively.

The main features currently available in Emukit are:

* **Multi-fidelity emulation:** build surrogate models when data is obtained from multiple information sources that have different fidelity and/or cost;
* **Bayesian optimisation:** optimise physical experiments and tune parameters of machine learning algorithms;
* **Experimental design/Active learning:** design the most informative experiments and perform active learning with machine learning models;
* **Sensitivity analysis:** analyse the influence of inputs on the outputs of a given system;
* **Bayesian quadrature:** efficiently compute the integrals of functions that are expensive to evaluate.

Emukit is agnostic to the underlying modelling framework, which means you can use any tool of your choice in the Python ecosystem to build the machine learning model, and still be able to use Emukit.

## Installation

To install emukit, simply run
```
pip install emukit
```

For other install options, see our [documentation](https://emukit.readthedocs.io/en/latest/installation.html).

### Dependencies / Prerequisites
Emukit's primary dependencies are Numpy and GPy.
See [requirements](requirements/requirements.txt).

## Getting started
For examples see our [tutorial notebooks](http://nbviewer.jupyter.org/github/emukit/emukit/blob/main/notebooks/index.ipynb).

## Documentation
To learn more about Emukit, refer to our [documentation](https://emukit.readthedocs.io).

To learn about emulation as a concept, check out the [Emukit playground](https://github.com/amzn/Emukit-playground) project.

## Citing the library

If you are using emukit, we would appreciate if you could cite our paper in your research:

    @inproceedings{emukit2019,
      author = {Paleyes, Andrei and Pullin, Mark and Mahsereci, Maren and Lawrence, Neil and González, Javier},
      title = {Emulation of physical processes with Emukit},
      booktitle = {Second Workshop on Machine Learning and the Physical Sciences, NeurIPS},
      year = {2019}
    }

## License

Emukit is licensed under Apache 2.0. Please refer to [LICENSE](LICENSE) and [NOTICE](NOTICE) for further license information.
