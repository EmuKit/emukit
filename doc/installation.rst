Installation
============

Emukit requires Python 3.5 or above and NumPy for basic functionality. Some core features also need GPy and GPyOpt. Some advanced elements may have their own dependencies, but their installation is optional.

Required dependecies can be installed from the ``requirements/requirements.txt`` file via

.. code-block:: bash

    pip install -r requirements/requirements.txt

Make sure ``pip`` refers to Python 3 installation.

After that, Emukit can be installed with

.. code-block:: bash

    git clone https://github.com/amzn/Emukit.git
    cd Emukit
    python setup.py install

or

.. code-block:: bash

    pip install git+https://github.com/amzn/Emukit.git