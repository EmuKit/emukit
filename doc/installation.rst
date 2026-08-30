Installation
============

Emukit requires Python 3.11 or above. To install the core emukit package, run

.. code-block:: bash

    pip install emukit

Optional dependencies
=====================
You can install optional dependency groups via setuptools extras. Each group enables additional functionality without inflating the core install:

- ``bnn``: Bayesian neural network (Bohamiann) and Profet meta-surrogate examples (adds ``pybnn``, ``torch``).
- ``sklearn``: scikit-learn model wrapper and related examples (adds ``scikit-learn``).
- ``docs``: Build documentation locally (Sphinx toolchain + GPy for rendering GP API docs).
- ``tests``: Test tooling only.
- ``examples``: Convenience bundle for most example scripts (installs ``GPy``, ``pybnn``, ``torch``, ``scikit-learn``).
- ``dev``: Convenience meta extra installing all of the above.

.. code-block:: bash
    # Bayesian neural network & Profet examples
    pip install emukit[bnn]

    # scikit-learn model wrapper
    pip install emukit[sklearn]

    # Build documentation locally
    pip install emukit[docs]

    # Bundle of example dependencies
    pip install emukit[examples]

    # Everything (core + bnn + sklearn + examples + docs + tests)
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
     # Or everything:
     pip install -e .[dev]

`python setup.py develop` is no longer needed; PEP 621 metadata in `pyproject.toml` enables editable installs directly via pip (PEP 660).
