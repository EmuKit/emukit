import pytest
import numpy as np
import random

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.sensitivity.monte_carlo import MonteCarloSensitivity
from emukit.sensitivity.test_functions import Ishigami


def test_montecarlo_sensitivity():

    ishigami = Ishigami(a=5, b=0.1)
    space = ParameterSpace([ContinuousParameter('x1', -np.pi, np.pi),
                            ContinuousParameter('x2', -np.pi, np.pi),
                            ContinuousParameter('x3', -np.pi, np.pi)])

    num_mc = 1000
    np.random.seed(0)
    senstivity_ishigami = MonteCarloSensitivity(ishigami.fidelity1, space)
    senstivity_ishigami.generate_samples(1)

    main_sample = np.array([[ 3.10732573, -1.25504469, -2.66820221], [-1.32105416, -1.45224686, -2.06642419],[ 2.41144646, -1.98949844,  1.66646315]])
    fixing_sample = np.array([[-2.93034587, -2.62148462,  1.71694805], [-2.70120457,  2.60061313,  3.00826754], [-2.56283615,  2.53817347, -1.00496868]])

    main_effects, total_effects, total_variance = senstivity_ishigami.compute_effects(main_sample = main_sample,fixing_sample=fixing_sample,num_monte_carlo_points = num_mc)

    keys = space.parameter_names

    assert(all(k in main_effects for k in keys))
    assert(all(k in total_effects for k in keys))
