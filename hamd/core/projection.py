"""
hamd.core.projection
====================
K-manifold projection helpers.

The K-cardinality constraint { x ∈ {0,1}^n : Σ x_i = K } defines a manifold.
HAMD enforces this during the continuous phase via:
  1. Top-K snap    — hard projection to nearest K-feasible binary point
  2. Mean-centering — projects gradient/velocity onto K-manifold tangent plane
"""

from __future__ import annotations

import numpy as np
import torch


def topk_snap(x: torch.Tensor, K: int) -> torch.Tensor:
    """
    Project a batch of continuous vectors to the K-feasible binary hypercube
    by setting the top-K entries to 1 and the rest to 0.

    Parameters
    ----------
    x : torch.Tensor, shape (batch, n), values in [0, 1]
    K : int, desired number of ones

    Returns
    -------
    torch.Tensor, shape (batch, n), binary {0, 1}
    """
    x_disc = torch.zeros_like(x)
    _, idx = torch.topk(x, K, dim=1)
    x_disc.scatter_(1, idx, 1.0)
    return x_disc


def k_project_grad(g: torch.Tensor) -> torch.Tensor:
    """
    Project a gradient (or any vector) onto the K-manifold tangent plane
    by subtracting its mean.  This ensures that gradient steps do not
    move the continuous trajectory off the equal-sum hyperplane.

    Parameters
    ----------
    g : torch.Tensor, shape (batch, n)

    Returns
    -------
    torch.Tensor, shape (batch, n), mean-centered along dim=1
    """
    return g - g.mean(dim=1, keepdim=True)


def topk_snap_np(x: np.ndarray, K: int) -> np.ndarray:
    """
    NumPy version of top-K snap for a single vector.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    K : int

    Returns
    -------
    np.ndarray, shape (n,), binary {0, 1}
    """
    out = np.zeros_like(x, dtype=np.int8)
    out[np.argpartition(x, -K)[-K:]] = 1
    return out
