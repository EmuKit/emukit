import pytest
 
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.experimental_design import RandomDesign
 
def test_random_design_returns_correct_number_of_points():
    p = ContinuousParameter('c', 1.0, 5.0)
    space = ParameterSpace([p])
    points_count = 5
 
    points = RandomDesign(space).get_samples(points_count)
 
    assert points_count == len(points)

