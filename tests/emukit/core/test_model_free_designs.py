from emukit.core import CategoricalParameter, ContinuousParameter, DiscreteParameter, ParameterSpace
from emukit.core.initial_designs import RandomDesign, LatinDesign


def create_model_free_designs(space: ParameterSpace):
    return [RandomDesign(space), LatinDesign(space)]


def test_design_returns_correct_number_of_points():
    p = ContinuousParameter('c', 1.0, 5.0)
    space = ParameterSpace([p])
    points_count = 5

    designs = create_model_free_designs(space)
    for design in designs:
        points = design.get_samples(points_count)

        assert points_count == len(points)
        columns_count = 1
        assert all([len(p) == columns_count for p in points])


def test_design_with_mixed_domain(encoding):
    p1 = ContinuousParameter('p1', 1.0, 5.0)
    p2 = CategoricalParameter('p2', encoding)
    p3 = DiscreteParameter('p3', [1, 2, 5, 6])
    space = ParameterSpace([p1, p2, p3])
    points_count = 5

    designs = create_model_free_designs(space)
    for design in designs:
        points = design.get_samples(points_count)

        assert points_count == len(points)
        # columns count is 1 for continuous plus 1 for discrete plus number of categories
        columns_count = 1 + 1 + len(encoding.categories)
        assert all([len(p) == columns_count for p in points])
