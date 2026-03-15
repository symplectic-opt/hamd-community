"""
Example: Small HUBO Instance
=============================
Demonstrates NativeCubicHAMD on a K-constrained cubic spin-glass problem
loaded from a pre-generated JSON file.

Run:
    python examples/example_hubo_small.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hamd import NativeCubicHAMD, load_instance
from hamd.baselines.sa_qubo import SAOnAugmentedQUBO
from hamd.core.utils import eval_cubic

DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'cubic_hubo', 'cubic_n50.json')

def main():
    inst   = load_instance(DATA_PATH)
    n      = inst['n']
    K      = inst.get('k_target', inst.get('K', n // 2))
    terms  = inst['cubic_terms']
    coeffs = inst['cubic_coeffs']
    Q_aug  = inst['Q_aug']

    print(f"Instance: cubic_n50   n={n}  K={K}  m={len(terms)}")
    print(f"  Augmented problem: n_aug={inst['n_aug']} "
          f"({inst['augmentation_ratio']:.1f}× inflation for SA/Tabu)\n")

    # HAMD — native cubic, hard K-constraint
    solver = NativeCubicHAMD(
        n=n, K=K,
        cubic_terms=terms, cubic_coeffs=coeffs,
        Q_quad=None,          # pure HUBO: no quadratic part
    )
    result = solver.solve(budget_sec=15.0, seed=42)
    print(f"HAMD  native_obj={result['best_value']:.4f}  "
          f"K={result['k_actual']}  restarts={result['n_restarts']}  "
          f"ils={result['n_ils']}  t={result['runtime_sec']:.1f}s")

    # SA — augmented QUBO
    sa = SAOnAugmentedQUBO(
        Q_aug=Q_aug, n_orig=n, K=K, seed=42, budget=15.0
    ).solve()
    sa_native = eval_cubic(sa['state'].astype(float), terms, coeffs)
    print(f"SA    native_obj={sa_native:.4f}  K={sa['k_actual']}  "
          f"t={sa['runtime_sec']:.1f}s")

    print(f"\nHAMD advantage: {result['best_value']:.4f} vs SA {sa_native:.4f} "
          f"({100*(sa_native - result['best_value'])/max(abs(sa_native),1e-9):.1f}% lower)")

if __name__ == '__main__':
    main()
