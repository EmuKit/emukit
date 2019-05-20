from hpolib.benchmarks.surrogates.svm import SurrogateSVM
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.examples.fabolas import fmin_fabolas

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


res = fmin_fabolas(wrapper, space=space, s_min=s_min, s_max=s_max, num_iters=100)
