"""
HAMD Community Edition — Small Cubic Portfolio Benchmark
=========================================================
Benchmarks HAMD-native vs SA-QUBO vs Tabu-QUBO on the cubic portfolio
instance cubicport_n200_k40.  This demonstrates the quadratization overhead
that SA and Tabu incur on higher-order portfolio objectives.

Usage
-----
    python benchmarks/run_portfolio_small.py
    python benchmarks/run_portfolio_small.py --budget-sec 30 --seeds 3
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
from hamd.core.metrics import gap_percent, print_benchmark_table, wintieloss
from hamd.core.utils import eval_native

SEED_BASE    = 42
DEFAULT_INST = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'cubic_portfolio',
    'cubicport_n200_k40.json')


def run(args: argparse.Namespace) -> None:
    inst_path = os.path.abspath(args.instance)
    if not os.path.exists(inst_path):
        print(f"[ERROR] Instance not found: {inst_path}")
        sys.exit(1)

    inst    = load_instance(inst_path)
    n       = inst['n']
    K       = inst['K']
    Q_quad  = inst['Q_quad']
    cterms  = inst['cubic_terms']
    ccoeffs = inst['cubic_coeffs']
    Q_aug   = inst['Q_aug']
    n_aug   = inst['n_aug']
    n_cubic = inst['n_cubic']

    print(f"\nInstance: {os.path.basename(inst_path)}")
    print(f"  n={n}  K={K}  n_cubic={n_cubic}  n_aug={n_aug}  "
          f"({inst['augmentation_ratio']:.1f}× augmentation)")
    print(f"  lambda_k={inst['lambda_k']:.2f}  "
          f"best_random_native={inst['best_random_native']:.4f}")

    hamd_vals, sa_vals, tabu_vals = [], [], []

    for seed_off in range(args.seeds):
        seed = SEED_BASE + seed_off * 1000
        print(f"\n{'─'*50}")
        print(f"Seed {seed}")

        # HAMD
        solver  = NativeCubicHAMD(
            n=n, K=K, Q_quad=Q_quad,
            cubic_terms=cterms, cubic_coeffs=ccoeffs,
        )
        res_h   = solver.solve(budget_sec=args.budget_sec, seed=seed)
        hamd_nat = res_h['best_value']
        hamd_vals.append(hamd_nat)
        print(f"  HAMD       native_obj={hamd_nat:.4f}  K={res_h['k_actual']}  "
              f"restarts={res_h['n_restarts']}  ils={res_h['n_ils']}  "
              f"t={res_h['runtime_sec']:.1f}s")

        # SA
        res_sa  = SAOnAugmentedQUBO(
            Q_aug=Q_aug, n_orig=n, K=K, seed=seed, budget=args.budget_sec
        ).solve()
        sa_nat  = eval_native(
            res_sa['state'].astype(np.float64), Q_quad, cterms, ccoeffs)
        sa_vals.append(sa_nat)
        print(f"  SA-QUBO    native_obj={sa_nat:.4f}  K={res_sa['k_actual']}  "
              f"t={res_sa['runtime_sec']:.1f}s  "
              f"gap={gap_percent(hamd_nat, sa_nat):+.1f}%")

        # Tabu
        res_tb  = TabuOnAugmentedQUBO(
            Q_aug=Q_aug, n_orig=n, K=K, seed=seed, budget=args.budget_sec
        ).solve()
        tb_nat  = eval_native(
            res_tb['state'].astype(np.float64), Q_quad, cterms, ccoeffs)
        tabu_vals.append(tb_nat)
        print(f"  Tabu-QUBO  native_obj={tb_nat:.4f}  K={res_tb['k_actual']}  "
              f"t={res_tb['runtime_sec']:.1f}s  "
              f"gap={gap_percent(hamd_nat, tb_nat):+.1f}%")

    if args.seeds > 1:
        print(f"\n{'='*50}")
        print(f"Multi-seed summary ({args.seeds} seeds):")
        print_benchmark_table(
            solvers=['HAMD', 'SA-AugQUBO', 'Tabu-AugQUBO'],
            results={
                'HAMD':        hamd_vals,
                'SA-AugQUBO':  sa_vals,
                'Tabu-AugQUBO': tabu_vals,
            },
            random_reference=inst['best_random_native'],
        )

    print(f"\n  Augmentation overhead: {n} → {n_aug} vars "
          f"(+{n_cubic} Rosenberg auxiliaries, {inst['augmentation_ratio']:.1f}×)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='HAMD cubic portfolio small benchmark')
    ap.add_argument('--instance',   default=DEFAULT_INST)
    ap.add_argument('--budget-sec', type=float, default=30.0)
    ap.add_argument('--seeds',      type=int,   default=1)
    run(ap.parse_args())
