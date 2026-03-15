"""
hamd.core.utils
===============
Shared objective-evaluation helpers and instance I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Objective evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_cubic(
    x: np.ndarray,
    terms: np.ndarray,
    coeffs: np.ndarray,
) -> float:
    """
    Evaluate a sparse degree-3 objective:
        H(x) = Σ_t  coeffs[t] · x[terms[t,0]] · x[terms[t,1]] · x[terms[t,2]]

    Parameters
    ----------
    x      : binary or continuous array, shape (n,)
    terms  : integer array, shape (m, 3)
    coeffs : float array,   shape (m,)

    Returns
    -------
    float  objective value
    """
    if len(terms) == 0:
        return 0.0
    a, b, c = terms[:, 0], terms[:, 1], terms[:, 2]
    return float(np.sum(coeffs * x[a] * x[b] * x[c]))


def eval_quadratic(x: np.ndarray, Q: np.ndarray) -> float:
    """Evaluate symmetric quadratic form x^T Q x."""
    return float(x @ Q @ x)


def eval_native(
    x: np.ndarray,
    Q_quad: np.ndarray,
    cubic_terms: np.ndarray,
    cubic_coeffs: np.ndarray,
) -> float:
    """
    Evaluate the full native cubic+quadratic objective.

        f(x) = x^T Q_quad x  +  Σ_t C_t · x_{a_t} x_{b_t} x_{c_t}

    Parameters
    ----------
    x            : binary vector, shape (n,)
    Q_quad       : quadratic cost matrix, shape (n, n) — symmetric
    cubic_terms  : integer array, shape (m, 3)
    cubic_coeffs : float array,   shape (m,)
    """
    return eval_quadratic(x, Q_quad) + eval_cubic(x, cubic_terms, cubic_coeffs)


# ─────────────────────────────────────────────────────────────────────────────
# Instance I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_instance(path: Union[str, Path]) -> dict:
    """
    Load a benchmark instance from a JSON file.

    Converts list fields to numpy arrays and returns a dict with keys:
        n, K, Q_quad, cubic_terms, cubic_coeffs, Q_aug, lambda_k, ...

    Parameters
    ----------
    path : path to the JSON instance file

    Returns
    -------
    dict with numpy arrays for matrix/vector fields
    """
    with open(path) as f:
        d = json.load(f)

    for key in ('cubic_terms',):
        if key in d:
            d[key] = np.array(d[key], dtype=np.int32)

    for key in ('cubic_coeffs',):
        if key in d:
            d[key] = np.array(d[key], dtype=np.float64)

    for key in ('Q_quad', 'Q_aug'):
        if key in d:
            d[key] = np.array(d[key], dtype=np.float64)

    return d
