"""
hamd.core.kswap
===============
K-swap steepest-descent polish for K-cardinality-constrained binary programs.

A K-swap move exchanges one variable currently set to 1 (going out) for one
currently set to 0 (coming in), preserving the exact cardinality K at every
step.  The inner loop finds and applies the best-improving swap until no
further improvement exists.

For a native cubic+quadratic objective the vectorised delta is:

    Δ_quad(i→0, j→1)  = -2Qx_i + 2Qx_j + Q_ii - 2Q_ij + Q_jj
    Δ_cubic(i→0, j→1) ≈ cubic_grad[j] - cubic_grad[i]   (first-order)

The first-order approximation is then verified with an exact forward evaluation
before accepting the swap, so no incorrect moves are accepted.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def kswap_polish(
    x_int: np.ndarray,
    Q_quad: np.ndarray,
    cubic_terms: np.ndarray,
    cubic_coeffs: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Best-improvement K-swap steepest descent on a native cubic+quadratic objective.

    Parameters
    ----------
    x_int        : initial binary vector, shape (n,), exactly K ones
    Q_quad       : symmetric quadratic cost matrix, shape (n, n)
    cubic_terms  : integer array, shape (m, 3)
    cubic_coeffs : float array, shape (m,)

    Returns
    -------
    (x_polished, objective_value)
        x_polished : improved binary vector with same K as input
        objective_value : f(x_polished)
    """
    x = x_int.copy().astype(np.float64)
    ones  = np.where(x > 0.5)[0]
    zeros = np.where(x < 0.5)[0]
    n     = len(x)

    Qx  = Q_quad @ x
    val = float(x @ Qx)

    def _cubic_grad(x_cur: np.ndarray) -> np.ndarray:
        g = np.zeros(n, dtype=np.float64)
        if len(cubic_terms) == 0:
            return g
        a, b, c = cubic_terms[:, 0], cubic_terms[:, 1], cubic_terms[:, 2]
        for pa, pb, pc in [(a, b, c), (b, a, c), (c, a, b)]:
            np.add.at(g, pa, cubic_coeffs * x_cur[pb] * x_cur[pc])
        return g

    # Add cubic part to current value
    if len(cubic_terms):
        a, b, c = cubic_terms[:, 0], cubic_terms[:, 1], cubic_terms[:, 2]
        val += float(np.sum(cubic_coeffs * x[a] * x[b] * x[c]))

    cg = _cubic_grad(x)
    Q_diag = np.diag(Q_quad)

    improved = True
    while improved:
        improved = False

        # Vectorised first-order delta for quadratic part: [|ones|, |zeros|]
        quad_delta = (
            - 2.0 * Qx[ones][:, None]
            + 2.0 * Qx[zeros][None, :]
            + Q_diag[ones][:, None]
            - 2.0 * Q_quad[np.ix_(ones, zeros)]
            + Q_diag[zeros][None, :]
        )
        # First-order cubic delta
        cubic_delta = cg[zeros][None, :] - cg[ones][:, None]
        total_delta = quad_delta + cubic_delta

        best_flat  = int(np.argmin(total_delta))
        best_delta = total_delta.flat[best_flat]

        if best_delta < -1e-9:
            bi = best_flat // len(zeros)
            bj = best_flat  % len(zeros)
            i_out, j_in = ones[bi], zeros[bj]

            # Exact verification
            x_try = x.copy()
            x_try[i_out] = 0.0
            x_try[j_in]  = 1.0
            new_val = float(x_try @ Q_quad @ x_try)
            if len(cubic_terms):
                a, b, c = cubic_terms[:, 0], cubic_terms[:, 1], cubic_terms[:, 2]
                new_val += float(np.sum(cubic_coeffs * x_try[a] * x_try[b] * x_try[c]))

            if new_val < val - 1e-9:
                x   = x_try
                val = new_val
                Qx  = Q_quad @ x
                cg  = _cubic_grad(x)
                ones  = np.where(x > 0.5)[0]
                zeros = np.where(x < 0.5)[0]
                improved = True

    return x.astype(np.int8), val
