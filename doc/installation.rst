Installation
============

Emukit requires Python 3.5 or above and NumPy for basic functionality. Some core features also need GPy and GPyOpt. Some advanced elements may have their own dependencies, but their installation is optional.

To install emukit, just run

.. code-block:: bash

    pip install emukit

Installation from sources
________
The simplest way to install emukit from sources is to run

.. code-block:: bash

    pip install git+https://github.com/amzn/Emukit.git

If you would like a bit more control, you can do it step by step: clone the repo, install dependencies, install emukit.

.. code-block:: bash

    git clone https://github.com/amzn/Emukit.git
    cd Emukit
    pip install -r requirements/requirements.txt
    python setup.py develop
