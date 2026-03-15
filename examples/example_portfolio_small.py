"""
Example: Cubic Portfolio (n=200)
==================================
Demonstrates NativeCubicHAMD on the cubic Markowitz portfolio problem
(n=200 assets, K=40 to select) vs SA and Tabu on the Rosenberg-augmented
QUBO.

Run:
    python examples/example_portfolio_small.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hamd import NativeCubicHAMD, load_instance
from hamd.baselines.sa_qubo import SAOnAugmentedQUBO
from hamd.baselines.tabu_qubo import TabuOnAugmentedQUBO
from hamd.core.metrics import gap_percent
from hamd.core.utils import eval_native

DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'cubic_portfolio',
    'cubicport_n200_k40.json')

def main():
    inst    = load_instance(DATA_PATH)
    n, K    = inst['n'], inst['K']
    Q_quad  = inst['Q_quad']
    cterms  = inst['cubic_terms']
    ccoeffs = inst['cubic_coeffs']
    Q_aug   = inst['Q_aug']

    print(f"Instance: cubicport_n200_k40")
    print(f"  n={n}  K={K}  n_cubic={inst['n_cubic']}  "
          f"n_aug={inst['n_aug']} ({inst['augmentation_ratio']:.1f}× augmentation)")
    print(f"  lambda_k={inst['lambda_k']:.0f}  random_ref={inst['best_random_native']:.4f}\n")

    # HAMD
    solver = NativeCubicHAMD(
        n=n, K=K, Q_quad=Q_quad,
        cubic_terms=cterms, cubic_coeffs=ccoeffs,
    )
    res_h = solver.solve(budget_sec=30.0, seed=42)
    print(f"HAMD  native_obj={res_h['best_value']:.4f}  K={res_h['k_actual']}  "
          f"restarts={res_h['n_restarts']}  ils={res_h['n_ils']}  "
          f"t={res_h['runtime_sec']:.1f}s")

    # SA
    res_sa = SAOnAugmentedQUBO(
        Q_aug=Q_aug, n_orig=n, K=K, seed=42, budget=30.0).solve()
    sa_nat = eval_native(res_sa['state'].astype(float), Q_quad, cterms, ccoeffs)
    print(f"SA    native_obj={sa_nat:.4f}  K={res_sa['k_actual']}  "
          f"t={res_sa['runtime_sec']:.1f}s  "
          f"gap={gap_percent(res_h['best_value'], sa_nat):+.1f}%")

    # Tabu
    res_tb = TabuOnAugmentedQUBO(
        Q_aug=Q_aug, n_orig=n, K=K, seed=42, budget=30.0).solve()
    tb_nat = eval_native(res_tb['state'].astype(float), Q_quad, cterms, ccoeffs)
    print(f"Tabu  native_obj={tb_nat:.4f}  K={res_tb['k_actual']}  "
          f"t={res_tb['runtime_sec']:.1f}s  "
          f"gap={gap_percent(res_h['best_value'], tb_nat):+.1f}%")

    print(f"\nAugmentation overhead: {n} → {inst['n_aug']} variables (+{inst['n_cubic']} aux).")
    print("HAMD operates on the native cubic landscape; SA/Tabu on the distorted QUBO.")

if __name__ == '__main__':
    main()
