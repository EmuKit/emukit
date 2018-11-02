Emukit Structure
============

.. contents::
    :local:

We have structured Emukit in a way that reflects our thinking about the different decision making processes it performs.
Bayesian optimization, experimental design and Bayesian quadrature are all decision making processes that follow a 
similar pattern. 
We think of all these decision making processes as implementations of a common abstract loop::

    while stopping condition is not met:
        optimize acquisition function
        evaluate user function
        update model with new observation

We have built Emukit in a modular way such that each fundamental component of this loop can be swapped out. 
If you are a machine learning researcher interested implemented in your method in Emukit - this is where to do find out
which parts you need to implement!

Loop
________
The ``emukit.core.loop.OuterLoop`` class is the abstract loop where the different components come together.
There are more specific loops for Bayesian optimization and experimental design that construct some of the component 
parts for you.


Model
________
All ``Emukit`` loops need a probabilistic model of the underlying system.
Emukit does not provide functionality to build models as there are already many good modelling frameworks available in python.
Instead, we provide a way of interfacing third part modelling libraries with Emukit. 
We already provide a wrapper for using a model created with ``GPy``.
For instructions on how to include your own model please see :doc:`this notebook </notebooks/Emukit-custom-model>`.

Different models and modelling frameworks will provide different functionality. 
For instance a Gaussian process will usually have derivatives of the predictions available but random forests will not. 
These different functionalities are represented by a set of interfaces which a model implements. 
The basic interface that all models must implement is ``IModel``, which implements functionality to make predictions and
update the model but a model may implement any number of other interfaces such as ``IDifferentiable`` which indicates a
model has prediction derivatives available.

Candidate Point Calculator
__________________________
This class decides which point to evaluate next.
The simplest implementation, ``Sequential``, collects one point at a time by finding where the acquisition is a maximum
by applying the acquisition optimizer to the acquisition function.
More complex implementations will enable batches of points to be collected so that the user function can be evaluated
in parallel.

Acquisition
___________
The acquisition is a heuristic quantification of how valuable collecting a future point might be.
It is used by the candidate point calculator to decide which point(s) to collect next.

Acquisition Optimizer
_____________________
The ``AcquisitionOptimizer`` optimizes the acquisition function to find the point at which the acquisition is a maximum.
This will use the acquisition function gradients if they are available. 
If gradients of the acquisition function are not available it will either estimate them numerically or use a gradient 
free optimizer.

User Function
_____________
This is the function that we are trying to reason about. 
It can be either evaluated by the user or it can be passed into the loop and evaluated by Emukit.

Model Updater
_____________
The ``ModelUpdater`` class updates the model with new training data after a new point is observed and optimizes any
hyper-parameters of the model. 
It can decide whether hyper-parameters need updating based on some internal logic.


Stopping Condition
__________________
The ``StoppingCondition`` class chooses when we should stop collecting points.
The most commonly used example is to stop when a set number of iterations have been reached.



