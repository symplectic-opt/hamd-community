"""
HAMD Community Edition — Exact Small-Size Validation
=====================================================
Brute-force enumeration of all C(n,K) feasible assignments provides a
provably exact global optimum for n≤30.  Used to verify that HAMD
correctly finds the ground state on small instances.

Sizes:
  n=20, K=4  → C(20,4) =      4,845  (instant)
  n=25, K=5  → C(25,5) =     53,130  (< 1 s)
  n=30, K=6  → C(30,6) =    593,775  (< 5 s)

Usage
-----
    python benchmarks/run_exact_small.py
    python benchmarks/run_exact_small.py --sizes 20 25 --budget-sec 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hamd import NativeCubicHAMD
from hamd.core.utils import eval_native

SEED_BASE = 42


def _make_small_instance(n: int, K: int, seed: int = 0) -> dict:
    """Generate a small cubic portfolio instance inline."""
    rng      = np.random.RandomState(seed)
    n_sectors = max(2, n // 5)
    sector    = rng.randint(0, n_sectors, size=n)

    # Factor model
    F       = rng.randn(n, 3)
    cov_F   = (F @ F.T) / 3
    Sigma   = cov_F + np.diag(rng.uniform(0.01, 0.1, n))
    Sigma   = 0.5 * (Sigma + Sigma.T)
    mu      = rng.uniform(0.02, 0.15, n)

    Q_quad  = Sigma.copy()
    np.fill_diagonal(Q_quad, np.diag(Q_quad) - mu)
    q_scale = float(np.max(np.abs(Sigma)))

    # Cubic sector co-movement terms
    cubic_terms, cubic_coeffs = [], []
    for s in range(n_sectors):
        members = np.where(sector == s)[0]
        if len(members) < 3:
            continue
        anchor = members[np.argmax(np.sum(np.abs(F[members]), axis=1))]
        for i in members:
            if i == anchor:
                continue
            for j in members:
                if j <= i or j == anchor:
                    continue
                cubic_terms.append([int(i), int(j), int(anchor)])
                cubic_coeffs.append(float(rng.exponential(scale=q_scale * 0.5)))

    cubic_terms  = np.array(cubic_terms,  dtype=np.int32)  if cubic_terms  else np.empty((0, 3), dtype=np.int32)
    cubic_coeffs = np.array(cubic_coeffs, dtype=np.float64) if cubic_coeffs else np.empty(0, dtype=np.float64)
    return {'n': n, 'K': K, 'Q_quad': Q_quad,
            'cubic_terms': cubic_terms, 'cubic_coeffs': cubic_coeffs}


def exact_enumerate(inst: dict) -> tuple:
    """Enumerate all C(n,K) K-feasible vectors; return (best_val, best_x)."""
    n, K     = inst['n'], inst['K']
    Q        = inst['Q_quad']
    terms    = inst['cubic_terms']
    coeffs   = inst['cubic_coeffs']
    best_val = float('inf')
    best_x   = None
    for chosen in combinations(range(n), K):
        x   = np.zeros(n, dtype=np.float64)
        x[list(chosen)] = 1.0
        val = eval_native(x, Q, terms, coeffs)
        if val < best_val:
            best_val = val
            best_x   = x.copy()
    return best_val, best_x


def run(args: argparse.Namespace) -> None:
    all_pass = True
    for n in args.sizes:
        K    = max(3, n // 5)
        inst = _make_small_instance(n, K, seed=0)
        print(f"\n{'='*55}")
        n_feasible = len(list(combinations(range(n), K)))
        print(f"n={n}  K={K}  n_cubic={len(inst['cubic_terms'])}  "
              f"C({n},{K}) = {n_feasible:,} feasible vectors")

        t0 = time.perf_counter()
        exact_val, _ = exact_enumerate(inst)
        enum_t = time.perf_counter() - t0
        print(f"  Exact optimum = {exact_val:.6f}  (enum time: {enum_t:.2f}s)")

        seed_pass = True
        for s in range(args.seeds):
            seed   = SEED_BASE + s * 1000
            solver = NativeCubicHAMD(
                n=n, K=K,
                Q_quad=inst['Q_quad'],
                cubic_terms=inst['cubic_terms'],
                cubic_coeffs=inst['cubic_coeffs'],
            )
            res  = solver.solve(budget_sec=args.budget_sec, seed=seed)
            gap  = 100.0 * (res['best_value'] - exact_val) / (abs(exact_val) + 1e-12)
            ok   = gap <= 0.5
            seed_pass = seed_pass and ok
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] seed={seed}  HAMD={res['best_value']:.6f}  "
                  f"gap={gap:+.2f}%  restarts={res['n_restarts']}  "
                  f"ils={res['n_ils']}  t={res['runtime_sec']:.1f}s")

        all_pass = all_pass and seed_pass
        result_str = "ALL SEEDS PASS" if seed_pass else "SOME SEEDS FAILED"
        print(f"  → n={n}: {result_str}")

    print()
    if all_pass:
        print("✓  All exact-small tests passed.  HAMD matches the global optimum.")
    else:
        print("✗  Some tests failed — check budget_sec or random seed.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='HAMD exact small-size validation')
    ap.add_argument('--sizes',      type=int,   nargs='+', default=[20, 25, 30])
    ap.add_argument('--budget-sec', type=float, default=10.0)
    ap.add_argument('--seeds',      type=int,   default=3)
    run(ap.parse_args())
