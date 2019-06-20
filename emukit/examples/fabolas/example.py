"""
This file implements an example of how to use Fabolas via the provided fmin interface to optimize the
hyperparameters of a SVM.
Instead of optimizing the real benchmark, we optimize a so-called surrogate benchmark instead which is much faster.
For more details look see th original Fabolas paper.

To run this example you need to install the new_benchmarks branch of HPOlib2:
    https://github.com/automl/HPOlib2/tree/new_benchmarks
"""
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.examples.fabolas import fmin_fabolas


try:
    from hpolib.benchmarks.surrogates.svm import SurrogateSVM
except ImportError:
    raise ImportError(
        'HPOLib is not installed. Please install it from: https://github.com/automl/HPOlib2/tree/master')

svm = SurrogateSVM()

l = []
for parameter in svm.get_configuration_space().get_hyperparameters():
    l.append(ContinuousParameter(parameter.name, parameter.lower, parameter.upper))

space = ParameterSpace(l)

s_min = 100
s_max = 50000


def wrapper(x, s):
    res = svm.objective_function(x, dataset_fraction=s/s_max)
    return res["function_value"], res["cost"]


res = fmin_fabolas(wrapper, space=space, s_min=s_min, s_max=s_max, n_iters=100, marginalize_hypers=False)
