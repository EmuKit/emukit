# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

from setuptools import setup, find_packages
import sys

from emukit.__version__ import __version__

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements/requirements.txt', 'r') as req:
    requires = req.read().split("\n")

# enforce >Python3 for all versions of pip/setuptools
assert sys.version_info >= (3,), 'This package requires Python 3.'

setup(
    name="emukit",
    version=__version__,
    description='Toolkit for decision making under uncertainty.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/emukit/emukit',
    packages=find_packages(exclude=['test*']),
    include_package_data=True,
    install_requires=requires,
    extras_require={'benchmarking': ['matplotlib']},
    python_requires='>=3',
    license='Apache License 2.0',
    classifiers=(
        # https://pypi.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    )
)
