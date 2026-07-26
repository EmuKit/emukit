Installation
============

Emukit requires Python 3.10 or above. The core installation provides the numerical stack and Emukit's pure numpy features. Gaussian process functionality based on GPy, multi-fidelity models, and quadrature models require the optional GPy extra.

To install the core emukit package (without GPy), run

.. code-block:: bash

    pip install emukit

Optional dependencies
=====================
You can install optional dependency groups via setuptools extras. Each group enables additional functionality without inflating the core install:

- ``gpy``: Gaussian process wrappers, multi-fidelity models, Bayesian quadrature (adds ``GPy``).
- ``bnn``: Bayesian neural network (Bohamiann) and Profet meta-surrogate examples (adds ``pybnn``, ``torch``).
- ``sklearn``: scikit-learn model wrapper and related examples (adds ``scikit-learn``).
- ``docs``: Build documentation locally (Sphinx toolchain + GPy for rendering GP API docs).
- ``tests``: Test tooling only.
- ``examples``: Convenience bundle for most example scripts (installs ``GPy``, ``pybnn``, ``torch``, ``scikit-learn``).
- ``dev``: Convenience meta extra installing all of the above.

.. code-block:: bash

    # Gaussian processes / multi-fidelity / quadrature
    pip install emukit[gpy]

    # Bayesian neural network & Profet examples
    pip install emukit[bnn]

    # scikit-learn model wrapper
    pip install emukit[sklearn]

    # Build documentation locally (includes gpy)
    pip install emukit[docs]

    # Bundle of example dependencies
    pip install emukit[examples]

    # Everything (gpy + bnn + sklearn + examples + docs + tests)
    pip install emukit[dev]
 
Installation from sources


.. code-block:: bash

    pip install git+https://github.com/emukit/emukit.git

If you would like a bit more control (e.g. for development), clone the repo, install dependencies, install emukit.

.. code-block:: bash

     git clone https://github.com/emukit/emukit.git
     cd Emukit
     # Editable install with desired extras (examples below)
     pip install -e .[tests]          # core + test tooling
     pip install -e .[gpy]            # add GPy-based functionality
     # Or everything:
     pip install -e .[dev]

`python setup.py develop` is no longer needed; PEP 621 metadata in `pyproject.toml` enables editable installs directly via pip (PEP 660).

NumPy 2 notice
===============
Core Emukit functionality works with NumPy 2.0+. However, some parts of Emukit (e.g. most acquisition functions) need GPy, that for the time being is a bit behind. If using GPy is critical for you, consider installing earlier versions of Emukit.