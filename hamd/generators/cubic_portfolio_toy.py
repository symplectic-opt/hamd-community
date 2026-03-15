"""
hamd.generators.cubic_portfolio_toy
=====================================
Generate small-to-medium cubic portfolio instances for research and evaluation.

Problem
-------
    minimise  f(x) = x^T Q_quad x  +  Σ_{(a,b,c) ∈ T}  C_{abc} x_a x_b x_c
    subject to  Σ_i x_i = K  (cardinality)
    x_i ∈ {0, 1}

Financial motivation
--------------------
Standard Markowitz portfolio risk (quadratic) is extended with cubic
third-order sector co-movement penalties, capturing co-skewness costs that
arise when three same-sector assets are held simultaneously. This requires
a genuine higher-order solver: classical QUBO methods must first
Rosenberg-quadratize, adding one auxiliary variable per cubic term and
distorting the optimisation landscape.

The community generator supports up to n=200.  For larger scales and
production-grade instance construction, see HAMD Enterprise Edition.

Usage
-----
    from hamd.generators.cubic_portfolio_toy import generate
    generate(n=100, K=20, filepath="data/cubic_portfolio/cubicport_n100_k20.json")
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


def _quadratize_portfolio(
    n: int,
    K: int,
    Q_quad: np.ndarray,
    cubic_terms: list,
    cubic_coeffs: list,
    lambda_rosenberg: float = 10.0,
    lambda_k: Optional[float] = None,
) -> tuple:
    """
    Rosenberg-quadratize the cubic portfolio objective for SA/Tabu comparison.

    Returns  (Q_aug, n_aug, lambda_k_used).
    """
    m     = len(cubic_terms)
    n_aug = n + m
    Q_aug = np.zeros((n_aug, n_aug), dtype=np.float64)
    Q_aug[:n, :n] = Q_quad

    for t_idx, ((a, b, c), coeff) in enumerate(zip(cubic_terms, cubic_coeffs)):
        w = n + t_idx
        R = lambda_rosenberg * abs(coeff)

        Q_aug[w, w] += 3.0 * R
        lo, hi = (a, b) if a < b else (b, a)
        Q_aug[lo, hi] += R
        lo, hi = (a, w) if a < w else (w, a)
        Q_aug[lo, hi] += -2.0 * R
        lo, hi = (b, w) if b < w else (w, b)
        Q_aug[lo, hi] += -2.0 * R
        lo, hi = (w, c) if w < c else (c, w)
        Q_aug[lo, hi] += coeff

    if lambda_k is None:
        max_abs  = max((abs(c) for c in cubic_coeffs), default=1.0)
        max_abs  = max(max_abs, float(np.max(np.abs(Q_quad))))
        lambda_k = max_abs * n * 4.0

    for i in range(n):
        Q_aug[i, i] += lambda_k * (1.0 - 2.0 * K)
    for i in range(n):
        for j in range(i + 1, n):
            Q_aug[i, j] += 2.0 * lambda_k

    return Q_aug, n_aug, lambda_k


def generate(
    n: int,
    K: int,
    filepath: str,
    seed: int = 42,
    alpha_cubic: float = 4.0,
    n_sectors: int = 10,
    risk_aversion: float = 1.0,
    return_scale: float = 1.5,
) -> None:
    """
    Generate a cubic portfolio instance and save to JSON.

    Parameters
    ----------
    n            : number of assets (max 200 for community edition)
    K            : portfolio cardinality (number of assets to select)
    filepath     : output path
    seed         : random seed
    alpha_cubic  : average cubic interaction edges per asset
    n_sectors    : number of market sectors
    risk_aversion: risk-aversion coefficient λ in λ·Σ - μ
    return_scale : scale of expected returns
    """
    if n > 200:
        raise ValueError(
            f"n={n} exceeds community edition limit (n≤200). "
            "Contact grserb.research@gmail.com for HAMD Enterprise Edition."
        )

    rng         = np.random.RandomState(seed)
    n_factors   = max(10, n // 5)
    sector_size = n // n_sectors

    # Factor-model covariance with sector structure
    L = rng.randn(n, n_factors) * 0.3
    sector_labels  = np.zeros(n, dtype=np.int32)
    base_returns   = rng.randn(n) * return_scale
    sector_anchors = []

    for s in range(n_sectors):
        lo = s * sector_size
        hi = min(lo + sector_size, n)
        sector_labels[lo:hi] = s
        sf = rng.randn(1, n_factors) * 0.5
        L[lo:hi, :] += sf
        base_returns[lo:hi] += rng.randn() * 2.0
        anchor = lo + int(np.argmax(np.abs(L[lo:hi, 0])))
        sector_anchors.append(anchor)

    cov    = L @ L.T + np.eye(n) * 0.5
    Q_quad = risk_aversion * cov.copy()
    for i in range(n):
        Q_quad[i, i] -= base_returns[i]
    Q_quad = 0.5 * (Q_quad + Q_quad.T)

    # Cubic terms: same-sector pairs × sector anchor
    n_cubic_target  = int(round(alpha_cubic * n))
    all_triples     = []
    for s in range(n_sectors):
        lo     = s * sector_size
        hi     = min(lo + sector_size, n)
        anchor = sector_anchors[s]
        members = [v for v in range(lo, hi) if v != anchor]
        for i in members:
            for j in members:
                if i < j:
                    all_triples.append((i, j, anchor))

    rng.shuffle(all_triples)
    triples = all_triples[:n_cubic_target]

    q_scale = float(np.mean(np.abs(np.diag(Q_quad))))
    coeffs  = rng.exponential(scale=q_scale * 0.5, size=len(triples))

    cubic_terms  = [list(t) for t in triples]
    cubic_coeffs = coeffs.tolist()

    Q_aug, n_aug, lk = _quadratize_portfolio(
        n=n, K=K, Q_quad=Q_quad,
        cubic_terms=cubic_terms, cubic_coeffs=cubic_coeffs)

    # Random reference (100 K-feasible samples)
    best_random = float('inf')
    for _ in range(100):
        x = np.zeros(n, dtype=np.float64)
        x[rng.choice(n, K, replace=False)] = 1
        val = float(x @ Q_quad @ x)
        if len(cubic_terms):
            ct = np.array(cubic_terms, dtype=np.int32)
            cc = np.array(cubic_coeffs)
            val += float(np.sum(cc * x[ct[:, 0]] * x[ct[:, 1]] * x[ct[:, 2]]))
        if val < best_random:
            best_random = val

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump({
            'n':                  n,
            'K':                  K,
            'n_cubic':            len(cubic_terms),
            'alpha_cubic':        alpha_cubic,
            'n_sectors':          n_sectors,
            'n_aug':              n_aug,
            'augmentation_ratio': round(n_aug / n, 2),
            'sector_labels':      sector_labels.tolist(),
            'sector_anchors':     sector_anchors,
            'Q_quad':             Q_quad.tolist(),
            'cubic_terms':        cubic_terms,
            'cubic_coeffs':       cubic_coeffs,
            'Q_aug':              Q_aug.tolist(),
            'lambda_rosenberg':   10.0,
            'lambda_k':           float(lk),
            'best_random_native': float(best_random),
        }, f)
    print(f"  Generated {filepath}  "
          f"(n={n}, K={K}, n_cubic={len(cubic_terms)}, n_aug={n_aug})")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate cubic portfolio instances (community sizes: n≤200)')
    parser.add_argument('--n',    type=int, default=100)
    parser.add_argument('--K',    type=int, default=None)
    parser.add_argument('--out',  default='data/cubic_portfolio/cubicport_n100_k20.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    K = args.K if args.K else max(5, args.n // 5)
    generate(n=args.n, K=K, filepath=args.out, seed=args.seed)
