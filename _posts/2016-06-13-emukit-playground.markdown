---
layout: post
title:  "Emukit-playground"
date:   2016-06-13 10:51:47 +0530
categories: jekyll update
img: city.png
categories: [one, two]
---
The Emukit-playground is a demo to illustrate different concepts in emulation an uncertainty quantification.

<div align="center"><img width="160" src="https://github.com/amzn/emukit-playground/raw/master/img/taxi.png" />        <img width="400" src="https://github.com/amzn/emukit-playground/raw/master/img/bayes.png" /></div>

<br/>

In uncertainty quantification, and emulator (also called surrogate model) is a name for an statistical model 
that is used in contexts where the problem is to predict the outputs of a complex (and usually deterministic) 
computer-based simulation model with some controllable inputs. It is generally the case that runs of the simulation model are computationally expensive. In general, some continuity in the variation of the outputs with respect to the inputs is assumed, which is necessary to enable statistical modelling.

An emulator is therefore the 'model of a model': it is an statistical model that aims to predict the unknown outputs of simulation model.

Emukit, has been designed with the idea of facilitate the used of emulators in decision loops. To illustrate these ideas 
alongside with Emukit we have released the [Emukit-playground](https://github.com/amzn/emukit-playground), an interactive learning tool developed by Adam Hirst 
for teaching users about emulation-based decision making. The playground allows users to train a working emulator in their browser based on visually driven simulations.


<p align="center"><a href="https://amzn.github.io/emukit-playground" class="btn btn-primary">Launch Playground</a>
<a href="https://github.com/amzn/emukit-playground" class="btn btn-success">Repo</a>
<a href="https://github.com/amzn/emukit-playground/blob/master/CONTRIBUTING.md" class="btn btn-info">Contribution guidelines</a></p>

We're always open to contributions! Please read our [contribution guidelines](CONTRIBUTING.md) for more information. We are particularly interested in contributions regarding translations and tutorials. This project is licensed under Apache 2.0. Please refer to [LICENSE](LICENSE) and [NOTICE](NOTICE) for further license information.


#### Refereces on Emulation 

- O'Hagan, A. (1978) [Curve fitting and optimal design for predictions](https://www.jstor.org/stable/2984861), *Journal of the Royal Statistical Society* B, 40, 1–42.
- O'Hagan, A. (2006) [Bayesian analysis of computer code outputs: A tutorial](https://www.sciencedirect.com/science/article/pii/S0951832005002383), *Reliability Engineering & System Safety*, 91, 1290–1300.