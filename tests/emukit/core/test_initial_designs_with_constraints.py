# Copyright 2020-2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.constraints import LinearInequalityConstraint, NonlinearInequalityConstraint
from emukit.core.initial_designs import RandomDesign
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.initial_designs.sobol_design import SobolDesign


def create_designs_with_space(space: ParameterSpace):
    """Create all design types for the given space."""
    return [RandomDesign(space), LatinDesign(space), SobolDesign(space)]


def test_designs_with_no_constraints():
    """Test that designs work normally when no constraints are present."""
    p1 = ContinuousParameter("p1", 0.0, 10.0)
    p2 = ContinuousParameter("p2", -5.0, 5.0)
    space = ParameterSpace([p1, p2])
    points_count = 10

    designs = create_designs_with_space(space)
    for design in designs:
        points = design.get_samples(points_count)
        assert points.shape == (points_count, 2)


def test_designs_respect_linear_inequality_constraints():
    """Test that designs respect linear inequality constraints."""
    p1 = ContinuousParameter("p1", 0.0, 10.0)
    p2 = ContinuousParameter("p2", 0.0, 10.0)
    
    # Constraint: p1 + p2 <= 12
    constraint = LinearInequalityConstraint(
        constraint_matrix=np.array([[1.0, 1.0]]),
        lower_bound=np.array([-np.inf]),
        upper_bound=np.array([12.0])
    )
    
    space = ParameterSpace([p1, p2], constraints=[constraint])
    points_count = 20

    designs = create_designs_with_space(space)
    for design in designs:
        points = design.get_samples(points_count)
        
        # Verify all points satisfy the constraint
        assert points.shape == (points_count, 2)
        constraint_values = points[:, 0] + points[:, 1]
        assert np.all(constraint_values <= 12.0 + 1e-6)  # Small tolerance for numerical errors


def test_designs_respect_nonlinear_constraints():
    """Test that designs respect nonlinear constraints."""
    p1 = ContinuousParameter("p1", 0.0, 5.0)
    p2 = ContinuousParameter("p2", 0.0, 5.0)
    
    # Constraint: p1^2 + p2^2 <= 16 (circle of radius 4)
    def circle_constraint(x):
        return x[:, 0] ** 2 + x[:, 1] ** 2
    
    constraint = NonlinearInequalityConstraint(
        constraint_function=circle_constraint,
        lower_bound=np.array([-np.inf]),
        upper_bound=np.array([16.0])
    )
    
    space = ParameterSpace([p1, p2], constraints=[constraint])
    points_count = 20

    designs = create_designs_with_space(space)
    for design in designs:
        points = design.get_samples(points_count)
        
        # Verify all points satisfy the constraint
        assert points.shape == (points_count, 2)
        constraint_values = circle_constraint(points)
        assert np.all(constraint_values <= 16.0 + 1e-6)  # Small tolerance for numerical errors


def test_designs_with_multiple_constraints():
    """Test that designs respect multiple constraints simultaneously."""
    p1 = ContinuousParameter("p1", 0.0, 10.0)
    p2 = ContinuousParameter("p2", 0.0, 10.0)
    
    # Constraint 1: p1 >= 2
    constraint1 = LinearInequalityConstraint(
        constraint_matrix=np.array([[1.0, 0.0]]),
        lower_bound=np.array([2.0]),
        upper_bound=np.array([np.inf])
    )
    
    # Constraint 2: p2 <= 8
    constraint2 = LinearInequalityConstraint(
        constraint_matrix=np.array([[0.0, 1.0]]),
        lower_bound=np.array([-np.inf]),
        upper_bound=np.array([8.0])
    )
    
    space = ParameterSpace([p1, p2], constraints=[constraint1, constraint2])
    points_count = 20

    designs = create_designs_with_space(space)
    for design in designs:
        points = design.get_samples(points_count)
        
        # Verify all points satisfy both constraints
        assert points.shape == (points_count, 2)
        assert np.all(points[:, 0] >= 2.0 - 1e-6)
        assert np.all(points[:, 1] <= 8.0 + 1e-6)


def test_design_fails_with_impossible_constraints():
    """Test that design raises error when constraints are impossible to satisfy."""
    p1 = ContinuousParameter("p1", 0.0, 5.0)
    
    # Constraint: p1 > 10 (impossible given bounds)
    constraint = LinearInequalityConstraint(
        constraint_matrix=np.array([[1.0]]),
        lower_bound=np.array([10.1]),
        upper_bound=np.array([np.inf])
    )
    
    space = ParameterSpace([p1], constraints=[constraint])
    
    designs = create_designs_with_space(space)
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
        upper_bound=np.array([20.0])
    )
    
    space = ParameterSpace([p1, p2], constraints=[constraint])
    
    # With low max_retries, should fail
    design_low_retries = RandomDesign(space, max_retries=1)
    with pytest.raises(RuntimeError, match="Could not generate"):
        design_low_retries.get_samples(5)
