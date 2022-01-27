import numpy as np
import pytest

from emukit.bayesian_optimization.acquisitions import MaxValueEntropySearch


@pytest.fixture
def max_value_entropy_search_acquisition(gpy_model, continuous_space):
    return MaxValueEntropySearch(gpy_model, continuous_space, num_samples = 10, grid_size = 5000)

def test_entropy_search_update_pmin(max_value_entropy_search_acquisition):
    max_value_entropy_search_acquisition.update_parameters()
    # check we sample required number of gumbel samplers
    assert max_value_entropy_search_acquisition.mins.shape[0] == max_value_entropy_search_acquisition.num_samples
