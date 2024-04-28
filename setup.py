# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import sys

from setuptools import find_packages, setup

from emukit.__version__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements/requirements.txt", "r") as req:
    requires = req.read().split("\n")

# enforce >Python3 for all versions of pip/setuptools
assert sys.version_info >= (3,), "This package requires Python 3."

setup(
    name="emukit",
    version=__version__,
    description="Toolkit for decision making under uncertainty.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emukit/emukit",
    packages=find_packages(exclude=["test*"]),
    include_package_data=True,
    install_requires=requires,
    extras_require={"benchmarking": ["matplotlib"]},
    python_requires=">=3.9",
    license="Apache License 2.0",
    classifiers=(
        # https://pypi.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ),
)
