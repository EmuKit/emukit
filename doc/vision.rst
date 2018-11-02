Emukit Vision
==============

.. contents::
    :local:

Preface about emulation
________________________


We see emulation comprising of three main parts:

- **Models**. This is a probabilistic data-driven representation of the process/simulator that the user is working with. There is normally a modelling framework that is used to create a model. Examples: neural network, Gaussian process, random forest. 
- **Methods**. Relatively low-level techniques that are aimed that either understanding, quantifying or using uncertainty that the model provides. Examples: Bayesian optimization, experimental design. 
- **Tasks**. High level goals that owners of the process/simulator might be actually interested in. Examples: measure quality of a simulator, explain complex system behavior. 

Typical workflow that we envision for a user interested in emulation is:

1. Figure out which questions/tasks are important for them in regard to their process/simulation.
2. Understand which emulation techniques are needed to accomplish the chosen task.
3. Build an emulator of the process. That can be a very involved step, that may include a lot of fine tuning and validation.
4. Feed the emulator to the chosen technique and use it to answer the question/complete the task. 

Emukit and emulation
_____________________

Here is Emukit approach towards each of three parts of emulation.

Methods
^^^^^^^^

This is the main focus of Emukit. Emukit defines a general sctructure of a decision making method, called ``OuterLoop``, and then offers implementations of few such methods: Bayesian optimization, sensitivity analysis, experimental design. All methods in Emukit are model-agnostic.

Models
^^^^^^^

Generally speaking, Emukit does not provide modelling capabilities, instead expecting users to bring their own models. Because of the variety of modelling frameworks out there, Emukit does not mandate or make any assumptions about a particular modelling technique or a library. Instead it suggests to implement a subset of defined model interfaces required to use a particular method. Nevertheless, there are a few model-related functionalities in Emukit:
- Example models, which give users something to play with to explore Emukit.
- Model wrappers, which are designed to help adapting models in particular modelling frameworks to Emukit interfaces.
- Multi-fidelity models, implemented based on `GPy <https://github.com/SheffieldML/GPy>`_.

Tasks
^^^^^^

Emukit does not contribute much to this part at the moment. However Emukit team are on lookuout for typical use cases for Emukit, and if a reoccuring pattern emerges, it may become a part of the library.
