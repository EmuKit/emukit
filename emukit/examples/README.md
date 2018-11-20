# MXNet Examples

This page contains a curated list of Emukit examples and links to other tutorials. It is heavily inspired on [contributing
examples](https://github.com/apache/incubator-mxnet/blob/master/example/README.md) section of [MXnet](https://mxnet.apache.org/).
 
  - [Contributing](#contributing)
  - [List of examples](#list-of-examples)
  - [List of tutorials](#list-of-tutorials)


## <a name="Contributing"></a>Contributing
------------------

If you want to contribute to this of, please open a new pull request.


### Examples

Examples can be either notebooks that tell a narrative about a problem/question studied with the elements of Emukit, *e.g* the analysis of the properties of a simulator, or
they can contain the implementation of some specific method. These can live in a .py file, should have tests 
and ideally should be delivered together with a small illustrative notebook. 

### Examples location

Example applications or scripts should be submitted in this `emukit/examples` folder.  Each example must live in a separated 
folder that can contain some extra files an dependencies.


#### Examples tests

As part of making sure all our examples are running correctly with the latest version of Emukit, yor can add your own tests 
here `tests/tutorials/test_tutorials.py`. (If you forget, don't worry your PR will not pass the sanity check).

### Tutorials

Tutorials are Jupyter notebooks that illustrate different features of the library. They are stand alone notebooks that 
don't require any extra file and fully sit on Emukit components (apart from the creation of the model).

If you have a tutorial idea, please download the [Jupyter notebook tutorial template](https://github.com/amzn/emukit/blob/develop/notebooks/Emukit-tutorial-how-to-write-a-notebook.ipynb).

#### Tutorial location

Notebook tutorials should be submitted in the `/notebooks` folder.

Do not forget to update the `notebooks/index.ipynb` for your tutorial to show up on the website.

## <a name="list-of-examples"></a>List of examples
------------------

* [Mountain car](https://github.com/amzn/emukit/tree/develop/emukit/examples/emulation_montain_car_simulator) - Optimization of the control policy of the mountain car simulator. Optimization is applied using an emulator of the reward and of the dynamics of the simulator.
* [Gaussian process Bayesian Optimization](https://github.com/amzn/emukit/tree/develop/emukit/examples/gp_bayesian_optimization) - Wrapper for using Bayesian optimization with Gaussian processes.
* [Modes](https://github.com/amzn/emukit/tree/develop/emukit/examples/models) - Implementation of a variety of models that can be used in combination with other Emukit features.
* [Cost sensitive Bayesian optimization](https://github.com/amzn/emukit/tree/develop/emukit/examples/cost_sensitive_bayesian_optimization) - Wrapper for using Bayesian optimization when there is a cost involved in the evaluation of the objective.


## <a name="list-of-tutorials"></a>List of tutorials
------------------
Visit the [index of tutorials](http://nbviewer.jupyter.org/github/amzn/emukit/blob/develop/notebooks/index.ipynb).