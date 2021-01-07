# Emukit Examples

This page contains a curated list of Emukit examples and links to other tutorials. It is heavily inspired by [contributing
examples](https://github.com/apache/incubator-mxnet/blob/master/example/README.md) section of [MXnet](https://mxnet.apache.org/).
 
  - [Contributing](#contributing)
  - [List of examples](#list-of-examples)
  - [List of tutorials](#list-of-tutorials)


## <a name="Contributing"></a>Contributing
If you want to contribute to this folder, please open a new pull request.


### Examples

Examples can be either notebooks that tell a story about a problem/question using Emukit, *e.g* the analysis of the properties of a simulator, or
they can contain an implementation of some specific method. Examples can live in a .py file, and ideally have tests and 
come with an illustrative notebook. 

#### Examples location

Example applications or scripts should be submitted in this `emukit/examples` folder.  Each example must live in a separated 
folder that can contain some extra files and dependencies. Please make sure that you update this `README.md` file with the information 
about you example when submitting a PR.


#### Examples tests

As part of making sure all our examples are running correctly with the latest version of Emukit, yor can add your own tests 
here `emukit/tests/examples/test_example.py` (if you forget, don't worry, we'll remind you during the review).

#### Tutorials

Tutorials are Jupyter notebooks that illustrate different features of the library. They are stand alone notebooks that 
don't require any extra file and fully sit on Emukit components (apart from the creation of the model).

If you have a tutorial idea, please download the [Jupyter notebook tutorial template](https://github.com/amzn/emukit/blob/master/notebooks/Emukit-tutorial-how-to-write-a-notebook.ipynb).

#### Tutorial location

Notebook tutorials should be submitted in the `/notebooks` folder.

Do not forget to update the `notebooks/index.ipynb` for your tutorial to show up on the website.

## <a name="list-of-examples"></a>List of examples

* [Gaussian process Bayesian Optimization](https://github.com/amzn/emukit/tree/master/emukit/examples/gp_bayesian_optimization) - Wrapper for using Bayesian optimization with Gaussian processes.
* [Vanilla Bayesian Quadrature](https://github.com/amzn/emukit/tree/master/emukit/examples/vanilla_bayesian_quadrature_with_rbf) - Wrapper for vanilla Bayesian quadrature that uses a Gaussian processes with an RBF kernel.
* [Models](https://github.com/amzn/emukit/tree/master/emukit/examples/models) - Implementation of a variety of models that can be used in combination with other Emukit features.
* [Mountain car](https://github.com/amzn/emukit/tree/master/emukit/examples/emulation_montain_car_simulator) - Optimization of the control policy of the mountain car simulator. Optimization is applied using an emulator of the reward and of the dynamics of the simulator.
* [Fabolas](https://github.com/EmuKit/emukit/tree/master/emukit/examples/fabolas) - Implementation of [Fabolas](https://arxiv.org/abs/1605.07079).
* [Preferential Batch Bayesian Optimization](https://github.com/EmuKit/emukit/tree/master/emukit/examples/preferential_batch_bayesian_optimization)- impementation of [PBBO method](https://arxiv.org/abs/2003.11435).
* [Profet](https://github.com/EmuKit/emukit/tree/master/emukit/examples/preferential_batch_bayesian_optimization) - impementation of [Meta-Surrogate Benchmarking](https://arxiv.org/abs/1905.12982).
* [SEIR Model](https://github.com/EmuKit/emukit/tree/master/emukit/examples/spread_of_disease-with_seir_model) - example of using Bayesian Quadrature to analyze behavior of the epidemic model.


## <a name="list-of-tutorials"></a>List of tutorials
Visit the [index of tutorials](http://nbviewer.jupyter.org/github/amzn/emukit/blob/master/notebooks/index.ipynb).
