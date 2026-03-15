"""
Tests for exact small-size validation.

Run:
    pytest tests/test_exact_small.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from itertools import combinations

import numpy as np
import pytest

from hamd import NativeCubicHAMD
from hamd.core.utils import eval_native


def _toy_instance(n=15, K=3, seed=0):
    rng = np.random.RandomState(seed)
    Q_quad = rng.randn(n, n)
    Q_quad = 0.5 * (Q_quad + Q_quad.T) * 0.1

    terms  = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    coeffs = np.array([0.5, -0.3], dtype=np.float64)

    if n < 4:
        terms  = np.empty((0, 3), dtype=np.int32)
        coeffs = np.empty(0, dtype=np.float64)

    return n, K, Q_quad, terms, coeffs


def exact_opt(n, K, Q_quad, terms, coeffs):
    best = float('inf')
    for chosen in combinations(range(n), K):
        x   = np.zeros(n, dtype=np.float64)
        x[list(chosen)] = 1.0
        val = eval_native(x, Q_quad, terms, coeffs)
        if val < best:
            best = val
    return best


@pytest.mark.parametrize("n,K", [(10, 2), (15, 3), (20, 4)])
def test_hamd_matches_exact(n, K):
    _, _, Q_quad, terms, coeffs = _toy_instance(n=n, K=K, seed=n)

    best_exact = exact_opt(n, K, Q_quad, terms, coeffs)
    solver = NativeCubicHAMD(
        n=n, K=K, Q_quad=Q_quad,
        cubic_terms=terms, cubic_coeffs=coeffs,
        batch_size=8,
    )
    res = solver.solve(budget_sec=5.0, seed=42)
    gap = 100.0 * (res['best_value'] - best_exact) / (abs(best_exact) + 1e-12)
    assert gap <= 2.0, f"n={n},K={K}: gap={gap:.2f}% > 2%"


@pytest.mark.parametrize("seed", [42, 1042, 2042])
def test_cardinality_exact(seed):
    n, K = 15, 3
    _, _, Q_quad, terms, coeffs = _toy_instance(n=n, K=K, seed=seed)
    solver = NativeCubicHAMD(n=n, K=K, Q_quad=Q_quad,
                              cubic_terms=terms, cubic_coeffs=coeffs)
    res = solver.solve(budget_sec=3.0, seed=seed)
    assert res['k_actual'] == K, f"K={K} but got k_actual={res['k_actual']}"


def test_pure_hubo_no_quadratic():
    """NativeCubicHAMD with Q_quad=None (pure HUBO)."""
    n, K = 20, 10
    rng    = np.random.RandomState(7)
    terms  = np.array([[i, (i+1) % n, (i+2) % n] for i in range(0, n, 3)], dtype=np.int32)
    coeffs = rng.choice([-1.0, 1.0], size=len(terms)).astype(np.float64)

    solver = NativeCubicHAMD(n=n, K=K, Q_quad=None,
                              cubic_terms=terms, cubic_coeffs=coeffs)
    res = solver.solve(budget_sec=3.0, seed=42)
    assert res['k_actual'] == K
    assert isinstance(res['best_value'], float)
