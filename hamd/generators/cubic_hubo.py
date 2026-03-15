"""
hamd.generators.cubic_hubo
===========================
Generate sparse K-constrained cubic spin-glass (HUBO) instances.

Problem
-------
    minimise  H(x) = Σ_{(a,b,c) ∈ E}  J_{abc} · x_a x_b x_c
    subject to  Σ_i x_i = K  (K = n // 2)
    x_i ∈ {0, 1}

Edge density:  m ≈ α · n  random triples,  J_{abc} ∈ {−1, +1}  uniformly.

The generated JSON includes both the native cubic representation and a
Rosenberg-quadratized augmented QUBO for SA/Tabu baselines.

Usage
-----
    python -m hamd.generators.cubic_hubo          # generates default sizes
    from hamd.generators.cubic_hubo import generate
    generate(n_vars=50, filepath="data/cubic_hubo/cubic_n50.json")
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Rosenberg quadratization helper
# ─────────────────────────────────────────────────────────────────────────────

def _quadratize(
    cubic_terms: list,
    cubic_coeffs: list,
    n: int,
    k_target: int,
    lambda_rosenberg: float = 10.0,
    lambda_k: Optional[float] = None,
) -> tuple:
    """
    Convert K-constrained cubic objective to an augmented QUBO via the
    Rosenberg reduction.

    Each cubic term  J·x_a x_b x_c  is replaced by:
      J·w·x_c  subject to  w = x_a x_b   (enforced by a quadratic penalty)

    where  w  is a new binary auxiliary variable.  The soft K-cardinality
    constraint is added as  λ_K (Σ x_i − K)².

    Returns
    -------
    Q_aug     : np.ndarray  [n_aug, n_aug]  augmented QUBO matrix
    n_aug     : int         n + len(cubic_terms)
    lambda_k  : float       K-penalty used
    """
    m     = len(cubic_terms)
    n_aug = n + m
    Q_aug = np.zeros((n_aug, n_aug), dtype=np.float64)

    for t_idx, ((a, b, c), coeff) in enumerate(zip(cubic_terms, cubic_coeffs)):
        w = n + t_idx                    # auxiliary variable index
        R = lambda_rosenberg * abs(coeff)

        # Rosenberg penalty: R·(3w + x_a x_b − 2x_a w − 2x_b w)
        Q_aug[w, w]   += 3.0 * R
        lo, hi = (a, b) if a < b else (b, a)
        Q_aug[lo, hi] += R
        lo, hi = (a, w) if a < w else (w, a)
        Q_aug[lo, hi] += -2.0 * R
        lo, hi = (b, w) if b < w else (w, b)
        Q_aug[lo, hi] += -2.0 * R

        # Cubic replaced by:  coeff · w · x_c
        lo, hi = (w, c) if w < c else (c, w)
        Q_aug[lo, hi] += coeff

    # Soft K-cardinality penalty:  λ_K (Σ x_i − K)²
    if lambda_k is None:
        max_abs  = max(abs(c) for c in cubic_coeffs) if cubic_coeffs else 1.0
        lambda_k = max_abs * n * 4.0
    for i in range(n):
        Q_aug[i, i] += lambda_k * (1.0 - 2.0 * k_target)
    for i in range(n):
        for j in range(i + 1, n):
            Q_aug[i, j] += 2.0 * lambda_k

    return Q_aug, n_aug, lambda_k


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    n_vars: int,
    filepath: str,
    seed: int = 42,
    alpha: float = 4.0,
) -> None:
    """
    Generate a sparse K-constrained cubic spin-glass instance and save to JSON.

    Parameters
    ----------
    n_vars   : number of binary variables
    filepath : output path
    seed     : random seed
    alpha    : average number of cubic edges per variable  (density)
    """
    rng      = np.random.RandomState(seed)
    k_target = n_vars // 2
    n_edges  = int(round(alpha * n_vars))

    # Sample random triples (no duplicates)
    seen    = set()
    triples = []
    while len(triples) < n_edges:
        t = tuple(sorted(rng.choice(n_vars, 3, replace=False).tolist()))
        if t not in seen:
            seen.add(t)
            triples.append(list(t))
    triples = np.array(triples, dtype=np.int32)   # [m, 3]
    coeffs  = rng.choice([-1.0, 1.0], size=len(triples)).astype(np.float64)

    # Quick greedy feasible reference (100 random K-solutions)
    best_random = float('inf')
    for _ in range(100):
        x = np.zeros(n_vars, dtype=np.int8)
        x[rng.choice(n_vars, k_target, replace=False)] = 1
        val = float(np.sum(coeffs * x[triples[:, 0]] * x[triples[:, 1]] * x[triples[:, 2]]))
        if val < best_random:
            best_random = val

    Q_aug, n_aug, lk = _quadratize(
        triples.tolist(), coeffs.tolist(), n_vars, k_target)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump({
            'n':              n_vars,
            'K':              k_target,
            'k_target':       k_target,
            'n_edges':        n_edges,
            'cubic_terms':    triples.tolist(),
            'cubic_coeffs':   coeffs.tolist(),
            'Q_aug':          Q_aug.tolist(),
            'n_aug':          n_aug,
            'lambda_k':       float(lk),
            'lambda_rosenberg': 10.0,
            'best_random_native': float(best_random),
            'augmentation_ratio': round(n_aug / n_vars, 2),
        }, f)
    print(f"  Generated {filepath}  "
          f"(n={n_vars}, K={k_target}, m={n_edges}, n_aug={n_aug})")


# ─────────────────────────────────────────────────────────────────────────────
# Default CLI: generate the three community data files
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SIZES: List[int] = [50, 75, 100]
DATA_DIR = "data/cubic_hubo"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate cubic HUBO instances')
    parser.add_argument('--sizes', type=int, nargs='+', default=DEFAULT_SIZES)
    parser.add_argument('--out-dir', default=DATA_DIR)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=4.0)
    args = parser.parse_args()

    print("Generating cubic HUBO instances …")
    for n in args.sizes:
        fp = os.path.join(args.out_dir, f"cubic_n{n}.json")
        generate(n_vars=n, filepath=fp, seed=args.seed, alpha=args.alpha)
    print("Done.")
