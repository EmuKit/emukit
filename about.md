---
layout: default
title: About
permalink: /about/
---

<h1>Emukit: the first open source toolkit for statistical emulation and decision making under uncertainty</h1>

*Cambridge, UK, December 2018.*

We are launching Emukit, a highly adaptable Python toolkit for enriching decision making under uncertainty. Emukit allows scientists to easily build and test their uncertainty quantification workflows.

Emukit is a highly adaptable Python toolkit for enriching decision making under uncertainty. This is particularly pertinent to complex systems where data is scarce or difficult to acquire. In these scenarios, propagating well-calibrated uncertainty estimates within a design loop or computational pipeline ensures that constrained resources are used effectively.

Emukit is agnostic to the underlying modelling framework, which means you can use any tool of your choice in the Python ecosystem to build the machine learning model, and still be able to use Emukit.


<h3> Sequential decison making under uncertainty</h3>

<div align="center"><img width="460" src="../images/loop.jpeg" />       </div>


<h3> One loop to run them all </h3>


<div align="center"><img width="460" src="../images/application_loops.jpeg" />      </div>


<h3> Build your model, run your method, solve your task </h3>


We see emulation comprising of three main parts:
- **Models**. This is a probabilistic data-driven representation of the process/simulator that the user is working with. There is normally a modelling framework that is used to create a model. Examples: neural network, Gaussian process, random forest. 
- **Methods**. Relatively low-level techniques that are aimed that either understanding, quantifying or using uncertainty that the model provides. Examples: Bayesian optimization, experimental design. 
- **Tasks**. High level goals that owners of the process/simulator might be actually interested in. Examples: measure quality of a simulator, explain complex system behavior. 

The typical workflow that we envision for a user interested in emulation is:
1. Figure out which questions/tasks are important for them in regard to their process/simulation.
2. Understand which emulation techniques are needed to accomplish the chosen task.
3. Build an emulator of the process. That can be a very involved step, that may include a lot of fine tuning and validation.
4. Feed the emulator to the chosen technique and use it to answer the question/complete the task. 

<div align="center"><img width="660" src="../images//model_method_task.jpeg" />       </div>



Here is Emukit approach towards each of three parts of emulation.

<h4> Models </h4>

 Generally speaking, Emukit does not provide modelling capabilities, instead expecting users to bring their own models. Because of the variety of modelling frameworks out there, Emukit does not mandate or make any assumptions about a particular modelling technique or a library. Instead it suggests to implement a subset of defined model interfaces required to use a particular method. Nevertheless, there are a few model-related functionalities in Emukit:
* **Example models**, which give users something to play with to explore Emukit.
* **Model wrappers**, which are designed to help adapting models in particular modelling frameworks to Emukit interfaces.
* **Multi-fidelity** models, implemented based on `GPy <https://github.com/SheffieldML/GPy>`_.

<h4> Methods </h4>

This is the main focus of Emukit. Emukit defines a general sctructure of a decision making method, called ``OuterLoop``, and then offers implementations of few such methods: Bayesian optimization, sensitivity analysis, experimental design. All methods in Emukit are model-agnostic.
 
 The main features currently available in Emukit are:

* **Bayesian optimisation:** optimise physical experiments and tune parameters of machine learning algorithms;
* **Experimental design/Active learning:** design the most informative experiments and perform active learning with machine learning models;
* **Sensitivity analysis:** analyse the influence of inputs on the outputs of a given system;
* **Bayesian quadrature:** efficiently compute the integrals of functions that are expensive to evaluate.
 
 
<h4> Tasks </h4>
 Emukit does not contribute much to this part at the moment. However Emukit team are on lookuout for typical use cases for Emukit, and if a reoccuring pattern emerges, it may become a part of the library.




<h3> What else? </h3>