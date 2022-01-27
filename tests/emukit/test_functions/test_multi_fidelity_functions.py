import pytest

from emukit.core.initial_designs import RandomDesign
from emukit.test_functions.multi_fidelity import (
    multi_fidelity_borehole_function,
    multi_fidelity_branin_function,
    multi_fidelity_currin_function,
    multi_fidelity_hartmann_3d,
    multi_fidelity_park_function,
)


@pytest.mark.parametrize('fcn', [multi_fidelity_borehole_function, multi_fidelity_branin_function,
                                 multi_fidelity_currin_function, multi_fidelity_hartmann_3d,
                                 multi_fidelity_park_function])
def test_multi_fidelity_function_shapes(fcn):
    n_points = 10
    fcn, space = fcn()
    random = RandomDesign(space)
    samples = random.get_samples(n_points)

    # There are only 2 or 3 fidelity functions in set of functions we are testing
    n_fidelities = len(space.parameters[-1].domain)
    if n_fidelities == 2:
        samples[:5, -1] = 0
        samples[5:, -1] = 1
    elif n_fidelities == 3:
        samples[:5, -1] = 0
        samples[5:8, -1] = 1
        samples[8:, -1] = 2
    else:
        raise ValueError('Please add a case for functions with {:.0f} fidelity levels'.format(n_fidelities))

    # Check shapes when calling through function wrapper
    results = fcn.evaluate(samples)
    assert len(results) == n_points
    for result in results:
        assert result.Y.shape == (1,)

    # Also check shape when calling each fidelity function individually
    for f in fcn.f:
        assert f(samples[:, :-1]).shape == (n_points, 1)
