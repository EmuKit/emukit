# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# These are notebooks that take too long to run so are not tested here
# The parallel evel notebook is too flaky and often fails the builds for no reason
excluded_notebooks = ['Emukit-tutorial-multi-fidelity-bayesian-optimization.ipynb',
                      'Emukit-tutorial-select-neural-net-hyperparameters.ipynb',
                      'Emukit-tutorial-parallel-eval-of-obj-fun.ipynb']

import os

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

notebook_directory = './notebooks/'


def get_notebook_names():
    """
    Get names of all notebooks in notebook directory
    :return:
    """
    extension = '.ipynb'

    notebook_names = []
    for file in os.listdir(notebook_directory):
        if file.endswith(extension) and (file not in excluded_notebooks):
            notebook_names.append(file)

    return notebook_names


@pytest.mark.parametrize("name", get_notebook_names())
def test_notebook_runs_without_errors(name):
    with open(os.path.join(notebook_directory, name)) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=120)
    ep.preprocess(nb, {'metadata': {'path': notebook_directory}})
