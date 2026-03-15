"""
Example: Exact Small-Size Validation
======================================
Enumerates all C(n,K) feasible solutions to find the provably exact global
optimum on a small cubic portfolio instance, then verifies HAMD matches it.

Run:
    python examples/example_exact_small.py
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from itertools import combinations
import numpy as np

from hamd import NativeCubicHAMD
from hamd.core.utils import eval_native


def make_instance(n=20, K=4, seed=42):
    rng = np.random.RandomState(seed)
    n_factors = 3
    F  = rng.randn(n, n_factors)
    Sigma = (F @ F.T) / n_factors + np.eye(n) * 0.1
    Sigma = 0.5 * (Sigma + Sigma.T)
    mu = rng.uniform(0.05, 0.20, n)
    Q_quad = Sigma.copy()
    np.fill_diagonal(Q_quad, np.diag(Q_quad) - mu)

    n_sectors = max(2, n // 5)
    sector = rng.randint(0, n_sectors, n)
    q_scale = float(np.max(np.abs(Sigma)))
    cubic_terms, cubic_coeffs = [], []
    for s in range(n_sectors):
        mems = np.where(sector == s)[0]
        if len(mems) < 3:
            continue
        anchor = mems[np.argmax(np.sum(np.abs(F[mems]), axis=1))]
        for i in mems:
            if i == anchor: continue
            for j in mems:
                if j <= i or j == anchor: continue
                cubic_terms.append([int(i), int(j), int(anchor)])
                cubic_coeffs.append(float(rng.exponential(q_scale * 0.5)))

    cubic_terms  = np.array(cubic_terms,  dtype=np.int32)  if cubic_terms  else np.empty((0,3), dtype=np.int32)
    cubic_coeffs = np.array(cubic_coeffs, dtype=np.float64) if cubic_coeffs else np.empty(0,    dtype=np.float64)
    return n, K, Q_quad, cubic_terms, cubic_coeffs


def main():
    n, K = 20, 4
    n, K, Q_quad, cubic_terms, cubic_coeffs = make_instance(n, K)

    print(f"Small cubic portfolio  n={n}  K={K}  n_cubic={len(cubic_terms)}")
    print(f"Enumerating all C({n},{K}) = {len(list(combinations(range(n),K))):,} feasible vectors...")

    t0 = time.perf_counter()
    best_exact = float('inf')
    for chosen in combinations(range(n), K):
        x = np.zeros(n, dtype=np.float64)
        x[list(chosen)] = 1.0
        v = eval_native(x, Q_quad, cubic_terms, cubic_coeffs)
        if v < best_exact:
            best_exact = v
    print(f"  Exact optimum = {best_exact:.6f}  (enum time: {time.perf_counter()-t0:.2f}s)\n")

    for seed in [42, 1042, 2042]:
        solver = NativeCubicHAMD(
            n=n, K=K, Q_quad=Q_quad,
            cubic_terms=cubic_terms, cubic_coeffs=cubic_coeffs,
        )
        res = solver.solve(budget_sec=5.0, seed=seed)
        gap = 100.0 * (res['best_value'] - best_exact) / (abs(best_exact) + 1e-12)
        status = "PASS" if gap <= 0.5 else "FAIL"
        print(f"  [{status}] seed={seed}  HAMD={res['best_value']:.6f}  gap={gap:+.2f}%")


if __name__ == '__main__':
    main()
