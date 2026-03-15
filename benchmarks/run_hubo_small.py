"""
HAMD Community Edition — Small HUBO Benchmark
==============================================
Benchmarks HAMD-native vs SA-QUBO vs Tabu-QUBO on K-constrained sparse
cubic spin-glass (HUBO) instances of sizes n=50, 75, 100.

All three solvers receive the same wall-clock budget.  HAMD operates on
the native n-variable cubic landscape with a hard K-constraint; SA and Tabu
operate on the Rosenberg-quadratized QUBO with n+m variables and a soft
K-penalty.

Usage
-----
    python benchmarks/run_hubo_small.py
    python benchmarks/run_hubo_small.py --budget-sec 30 --seeds 1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hamd import NativeCubicHAMD, load_instance
from hamd.baselines.sa_qubo import SAOnAugmentedQUBO
from hamd.baselines.tabu_qubo import TabuOnAugmentedQUBO
from hamd.core.metrics import print_benchmark_table
from hamd.core.utils import eval_cubic

SEED_BASE = 42
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'cubic_hubo')


def run(args: argparse.Namespace) -> None:
    data_dir = os.path.abspath(DATA_DIR)
    files    = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))

    if not files:
        print(f"[ERROR] No instances found in {data_dir}")
        print("  Run:  python -m hamd.generators.cubic_hubo  to regenerate.")
        sys.exit(1)

    for fname in files:
        path = os.path.join(data_dir, fname)
        inst = load_instance(path)
        n    = inst['n']
        K    = inst.get('k_target', inst.get('K', n // 2))
        terms   = inst['cubic_terms']
        coeffs  = inst['cubic_coeffs']
        Q_aug   = inst['Q_aug']
        n_aug   = inst['n_aug']

        print(f"\n{'='*60}")
        print(f"  {fname}   n={n}  K={K}  m={inst.get('n_edges', len(terms))}  "
              f"n_aug={n_aug}  ({n_aug/n:.1f}× augmentation)")
        print(f"{'='*60}")

        hamd_vals, sa_vals, tabu_vals = [], [], []

        for seed_off in range(args.seeds):
            seed = SEED_BASE + seed_off * 1000
            print(f"\n  Seed {seed}")

            # HAMD — native cubic, hard K-constraint
            solver = NativeCubicHAMD(
                n=n, K=K,
                cubic_terms=terms, cubic_coeffs=coeffs,
                Q_quad=None,
            )
            res_h = solver.solve(budget_sec=args.budget_sec, seed=seed)
            hamd_vals.append(float(res_h['best_value']))
            print(f"    HAMD      best={res_h['best_value']:.3f}  "
                  f"K={res_h['k_actual']}  restarts={res_h['n_restarts']}  "
                  f"ils={res_h['n_ils']}  t={res_h['runtime_sec']:.1f}s")

            # SA — augmented QUBO
            res_sa = SAOnAugmentedQUBO(
                Q_aug=Q_aug, n_orig=n, K=K, seed=seed, budget=args.budget_sec
            ).solve()
            # Translate QUBO objective → native cubic value
            sa_native = eval_cubic(res_sa['state'].astype(np.float64), terms, coeffs)
            sa_vals.append(sa_native)
            print(f"    SA-QUBO   native_obj={sa_native:.3f}  "
                  f"K={res_sa['k_actual']}  t={res_sa['runtime_sec']:.1f}s")

            # Tabu — augmented QUBO
            res_tb = TabuOnAugmentedQUBO(
                Q_aug=Q_aug, n_orig=n, K=K, seed=seed, budget=args.budget_sec
            ).solve()
            tb_native = eval_cubic(res_tb['state'].astype(np.float64), terms, coeffs)
            tabu_vals.append(tb_native)
            print(f"    Tabu-QUBO native_obj={tb_native:.3f}  "
                  f"K={res_tb['k_actual']}  t={res_tb['runtime_sec']:.1f}s")

        if args.seeds > 1:
            print(f"\n  Summary across {args.seeds} seeds (native cubic objective):")
            print_benchmark_table(
                solvers=['HAMD', 'SA-QUBO', 'Tabu-QUBO'],
                results={
                    'HAMD':     hamd_vals,
                    'SA-QUBO':  sa_vals,
                    'Tabu-QUBO': tabu_vals,
                },
                random_reference=inst.get('best_random_native'),
            )
        print(f"\n  Note: SA/Tabu search space is {n_aug} vars ({n}+{len(terms)} aux). "
              f"HAMD uses {n} vars with exact K-manifold constraint.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='HAMD HUBO small benchmark')
    ap.add_argument('--budget-sec', type=float, default=30.0)
    ap.add_argument('--seeds',      type=int,   default=1)
    run(ap.parse_args())
