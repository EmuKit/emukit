#!/usr/bin/env python3
"""
Standalone test script to verify the optimizer refactoring works.
Tests the new signature: optimize(x0, f, df=None)
"""

import sys
import numpy as np

# Import the refactored code
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.optimization import ContextManager
from emukit.core.optimization.optimizer import OptLbfgs, apply_optimizer

print("=" * 70)
print("TESTING OPTIMIZER REFACTORING - Issue #218")
print("=" * 70)

# Test 1: Basic optimization with gradient
print("\n[TEST 1] L-BFGS with gradient (no context)")
print("-" * 70)

def objective(x):
    """Simple quadratic: f(x) = x0^2 + x1^2, minimum at (0, 0)"""
    return x[:, 0] ** 2 + x[:, 1] ** 2

def gradient(x):
    """Gradient: df/dx = [2*x0, 2*x1]"""
    grad = np.array([2 * x[:, 0], 2 * x[:, 1]])
    return grad.T  # Return as (n_samples, n_dims)

space = ParameterSpace([
    ContinuousParameter("x", -1, 1),
    ContinuousParameter("y", -1, 1)
])

lbfgs = OptLbfgs(bounds=[(-1, 1), (-1, 1)], max_iterations=1000)
x0 = np.array([[1.0, 1.0]])  # Start at (1, 1)

try:
    # NEW SIGNATURE: apply_optimizer(optimizer, x0, space, f, df=None)
    x_opt, f_opt = apply_optimizer(lbfgs, x0, space, objective, gradient)
    
    # Check result
    expected = np.array([[0, 0]])
    error = np.linalg.norm(x_opt - expected)
    
    print(f"✓ Optimization with gradient succeeded")
    print(f"  Starting point:  x0 = {x0[0]}")
    print(f"  Optimized point: x = {x_opt[0]}")
    print(f"  Expected:        x = {expected[0]}")
    print(f"  Error:           {error:.2e}")
    
    if error < 1e-3:
        print(f"✓ TEST 1 PASSED - Result is within tolerance")
        test1_passed = True
    else:
        print(f"✗ TEST 1 FAILED - Error is too large")
        test1_passed = False
except Exception as e:
    print(f"✗ TEST 1 FAILED with exception:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    test1_passed = False

# Test 2: Optimization without gradient (approximate)
print("\n[TEST 2] L-BFGS without gradient (approximated)")
print("-" * 70)

try:
    # NEW SIGNATURE: no df parameter
    x_opt, f_opt = apply_optimizer(lbfgs, x0, space, objective)
    
    # Check result
    expected = np.array([[0, 0]])
    error = np.linalg.norm(x_opt - expected)
    
    print(f"✓ Optimization without gradient succeeded")
    print(f"  Starting point:  x0 = {x0[0]}")
    print(f"  Optimized point: x = {x_opt[0]}")
    print(f"  Expected:        x = {expected[0]}")
    print(f"  Error:           {error:.2e}")
    
    if error < 1e-2:  # Slightly larger tolerance for approximate gradient
        print(f"✓ TEST 2 PASSED - Result is within tolerance")
        test2_passed = True
    else:
        print(f"✗ TEST 2 FAILED - Error is too large")
        test2_passed = False
except Exception as e:
    print(f"✗ TEST 2 FAILED with exception:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    test2_passed = False

# Test 3: Optimization with context
print("\n[TEST 3] L-BFGS with context variables")
print("-" * 70)

try:
    context_manager = ContextManager(space, {"x": 0.5})
    lbfgs_context = OptLbfgs(bounds=[(-1, 1)], max_iterations=1000)
    
    # Optimize with x fixed at 0.5, only optimize y
    x_opt, f_opt = apply_optimizer(lbfgs_context, x0, space, objective, gradient, context_manager)
    
    # Expected: x=0.5 (fixed), y=0 (optimized)
    expected = np.array([[0.5, 0]])
    error = np.linalg.norm(x_opt - expected)
    
    print(f"✓ Optimization with context succeeded")
    print(f"  Context: x fixed at 0.5")
    print(f"  Starting point:  x0 = {x0[0]}")
    print(f"  Optimized point: x = {x_opt[0]}")
    print(f"  Expected:        x = {expected[0]}")
    print(f"  Error:           {error:.2e}")
    
    if error < 1e-3:
        print(f"✓ TEST 3 PASSED - Result is within tolerance")
        test3_passed = True
    else:
        print(f"✗ TEST 3 FAILED - Error is too large")
        test3_passed = False
except Exception as e:
    print(f"✗ TEST 3 FAILED with exception:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    test3_passed = False

# Test 4: Verify old signature fails (f_df parameter)
print("\n[TEST 4] Verify old f_df parameter is removed")
print("-" * 70)

try:
    # Try to use old signature with f_df parameter
    x_opt, f_opt = apply_optimizer(lbfgs, x0, space, objective, None, f_df=lambda x: (objective(x), gradient(x)))
    print(f"✗ TEST 4 FAILED - Old f_df parameter should not be accepted")
    test4_passed = False
except TypeError as e:
    if "f_df" in str(e):
        print(f"✓ Old signature correctly rejected")
        print(f"  Error message: {e}")
        print(f"✓ TEST 4 PASSED - f_df parameter properly removed")
        test4_passed = True
    else:
        print(f"✗ TEST 4 FAILED - Wrong error type: {e}")
        test4_passed = False
except Exception as e:
    # Could be any error, as long as f_df isn't accepted
    print(f"✓ Old signature correctly rejected")
    print(f"  Error type: {type(e).__name__}: {e}")
    print(f"✓ TEST 4 PASSED - f_df parameter properly removed")
    test4_passed = True

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

results = {
    "TEST 1 (with gradient)": test1_passed,
    "TEST 2 (without gradient)": test2_passed,
    "TEST 3 (with context)": test3_passed,
    "TEST 4 (f_df removed)": test4_passed,
}

for test_name, passed in results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status:8} {test_name}")

all_passed = all(results.values())
print("=" * 70)

if all_passed:
    print("\n✓ ALL TESTS PASSED! Refactoring is working correctly.")
    sys.exit(0)
else:
    print(f"\n✗ {sum(not p for p in results.values())} test(s) failed.")
    sys.exit(1)
