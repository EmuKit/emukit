Usage
============

.. contents::
   :local:

Logging
________

Emukit uses `logging module <https://docs.python.org/3/library/logging.html>`_ from the standard Python library to emit its logging information. However Emukit does not configure logging in any way, to make sure users of the library can setup logging to best suit their needs.

That means normally users won't see any logs on or below `INFO` level. This can be an issue if Emukit's execution runs for some time, as there will be no progress reported until the task is finished. If you would like to enable more verbosity to monitor progress, use this code to enable logging before calling Emukit:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.INFO)

Of course more advanced configuration is possible. We refer users to `official Python documentation <https://docs.python.org/3/howto/logging-cookbook.html>`_ for more information on logging setup.