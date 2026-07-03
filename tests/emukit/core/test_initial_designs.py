# Copyright 2020-2026 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from emukit.core import CategoricalParameter, ContinuousParameter, DiscreteParameter, ParameterSpace
from emukit.core.constraints import LinearInequalityConstraint, NonlinearInequalityConstraint
from emukit.core.initial_designs import RandomDesign
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.initial_designs.sobol_design import SobolDesign


def create_initial_designs(space: ParameterSpace):
    return [RandomDesign(space), LatinDesign(space), SobolDesign(space)]


def test_design_returns_correct_number_of_points():
    p = ContinuousParameter("c", 1.0, 5.0)
    space = ParameterSpace([p])
    points_count = 5

    designs = create_initial_designs(space)
    for design in designs:
        points = design.get_samples(points_count)

        assert points_count == len(points)
        assert all([len(p) == 1 for p in points])


def test_design_returns_points_within_bounds():
    p1 = ContinuousParameter("p1", 0.01, 0.05)
    p2 = ContinuousParameter("p2", -100.0, -90.0)
    space = ParameterSpace([p1, p2])
    points_count = 5

    designs = create_initial_designs(space)
    for design in designs:
        points = design.get_samples(points_count)

        for i, p in enumerate(space.parameters):
            assert np.all(p.min <= points[:, i])
            assert np.all(points[:, i] <= p.max)


def test_design_with_mixed_domain(encoding):
    p1 = ContinuousParameter("p1", 1.0, 5.0)
    p2 = CategoricalParameter("p2", encoding)
    p3 = DiscreteParameter("p3", [1, 2, 5, 6])
    space = ParameterSpace([p1, p2, p3])
    points_count = 5

    designs = create_initial_designs(space)
    for design in designs:
        points = design.get_samples(points_count)

        assert points_count == len(points)
        # columns count is 1 for continuous plus 1 for discrete plus number of categories
        columns_count = 1 + 1 + len(encoding.categories)
        assert all([len(p) == columns_count for p in points])


# Tests for constraint-respecting designs


def test_designs_respect_linear_inequality_constraints():
    """Test that designs respect linear inequality constraints."""
    p1 = ContinuousParameter("p1", 0.0, 10.0)
    p2 = ContinuousParameter("p2", 0.0, 10.0)

    # Constraint: p1 + p2 <= 18 (loose enough to be achievable)
    constraint = LinearInequalityConstraint(
        constraint_matrix=np.array([[1.0, 1.0]]), lower_bound=np.array([-np.inf]), upper_bound=np.array([18.0])
    )

    space = ParameterSpace([p1, p2], constraints=[constraint])
    points_count = 10

    designs = create_initial_designs(space)
    for design in designs:
        points = design.get_samples(points_count)

        # Verify all points satisfy the constraint
        assert points.shape == (points_count, 2)
        constraint_values = points[:, 0] + points[:, 1]
        assert np.all(constraint_values <= 18.0 + 1e-6)  # Small tolerance for numerical errors


def test_designs_respect_nonlinear_constraints():
    """Test that designs respect nonlinear constraints."""
    p1 = ContinuousParameter("p1", 0.0, 5.0)
    p2 = ContinuousParameter("p2", 0.0, 5.0)

    # Constraint: p1^2 + p2^2 <= 22 (circle of radius ~4.69, covers ~75% of space)
    # Note: constraint function receives a 1-d array (single point), not 2-d
    def circle_constraint(x):
        return x[0] ** 2 + x[1] ** 2

    constraint = NonlinearInequalityConstraint(
        constraint_function=circle_constraint, lower_bound=np.array([-np.inf]), upper_bound=np.array([22.0])
    )

    space = ParameterSpace([p1, p2], constraints=[constraint])
    points_count = 5

    designs = create_initial_designs(space)
    for design in designs:
        points = design.get_samples(points_count)

        # Verify all points satisfy the constraint
        assert points.shape == (points_count, 2)
        constraint_values = np.array([circle_constraint(p) for p in points])
        assert np.all(constraint_values <= 22.0 + 1e-6)  # Small tolerance for numerical errors


def test_designs_with_multiple_constraints():
    """Test that designs respect multiple constraints simultaneously."""
    p1 = ContinuousParameter("p1", 0.0, 10.0)
    p2 = ContinuousParameter("p2", 0.0, 10.0)

    # Constraint 1: p1 >= 0.5 (loose constraint, 95% of space)
    constraint1 = LinearInequalityConstraint(
        constraint_matrix=np.array([[1.0, 0.0]]), lower_bound=np.array([0.5]), upper_bound=np.array([np.inf])
    )

    # Constraint 2: p2 <= 9.5 (loose constraint, 95% of space)
    constraint2 = LinearInequalityConstraint(
        constraint_matrix=np.array([[0.0, 1.0]]), lower_bound=np.array([-np.inf]), upper_bound=np.array([9.5])
    )

    space = ParameterSpace([p1, p2], constraints=[constraint1, constraint2])
    points_count = 5

    designs = create_initial_designs(space)
    for design in designs:
        points = design.get_samples(points_count)

        # Verify all points satisfy both constraints
        assert points.shape == (points_count, 2)
        assert np.all(points[:, 0] >= 0.5 - 1e-6)
        assert np.all(points[:, 1] <= 9.5 + 1e-6)


def test_design_fails_with_impossible_constraints():
    """Test that design raises error when constraints are impossible to satisfy."""
    p1 = ContinuousParameter("p1", 0.0, 5.0)

    # Constraint: p1 > 10 (impossible given bounds)
    constraint = LinearInequalityConstraint(
        constraint_matrix=np.array([[1.0]]), lower_bound=np.array([10.0]), upper_bound=np.array([np.inf])
    )

    space = ParameterSpace([p1], constraints=[constraint])

    designs = create_initial_designs(space)
    for design in designs:
        with pytest.raises(RuntimeError, match="Could not generate"):
            design.get_samples(10)


def test_design_respects_max_retries():
    """Test that max_retries parameter controls retry behavior."""
    p1 = ContinuousParameter("p1", 0.0, 10.0)
    p2 = ContinuousParameter("p2", 0.0, 10.0)

    # Very restrictive constraint that's hard to satisfy
    constraint = LinearInequalityConstraint(
        constraint_matrix=np.array([[1.0, 1.0]]),
        lower_bound=np.array([19.5]),  # Very close to maximum
        upper_bound=np.array([20.0]),
    )

    space = ParameterSpace([p1, p2], constraints=[constraint])

    # Test with all design types
    designs = [
        RandomDesign(space),
        LatinDesign(space),
        SobolDesign(space),
    ]
    for design in designs:
        with pytest.raises(RuntimeError, match="Could not generate"):
            design.get_samples(5, max_retries=10)
