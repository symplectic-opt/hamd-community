"""
Tests for K-manifold projection utilities.

Run:
    pytest tests/test_projection.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import torch

from hamd.core.projection import topk_snap, topk_snap_np, k_project_grad
from hamd.core.kswap import kswap_polish


def test_topk_snap_binary():
    x = torch.tensor([[0.9, 0.3, 0.8, 0.1, 0.5]], dtype=torch.float64)
    snap = topk_snap(x, K=2)
    assert snap.sum().item() == 2.0
    assert snap[0, 0].item() == 1.0  # 0.9 is top
    assert snap[0, 2].item() == 1.0  # 0.8 is 2nd


def test_topk_snap_np():
    x = np.array([0.1, 0.9, 0.4, 0.8, 0.2])
    snap = topk_snap_np(x, K=2)
    assert int(snap.sum()) == 2
    assert snap[1] == 1
    assert snap[3] == 1


def test_k_project_mean_zero():
    g = torch.randn(5, 10, dtype=torch.float64)
    pg = k_project_grad(g)
    means = pg.mean(dim=1)
    assert torch.all(means.abs() < 1e-12), "Projected gradient should have zero mean"


@pytest.mark.parametrize("n,K", [(10, 3), (20, 5)])
def test_kswap_polish_preserves_cardinality(n, K):
    rng = np.random.RandomState(0)
    Q   = rng.randn(n, n); Q = 0.5 * (Q + Q.T)
    x   = np.zeros(n, dtype=np.int8)
    x[rng.choice(n, K, replace=False)] = 1
    terms  = np.empty((0, 3), dtype=np.int32)
    coeffs = np.empty(0, dtype=np.float64)
    x_pol, val = kswap_polish(x, Q, terms, coeffs)
    assert int(x_pol.sum()) == K


def test_kswap_improves_or_equal():
    rng = np.random.RandomState(1)
    n, K = 15, 5
    Q    = rng.randn(n, n); Q = 0.5 * (Q + Q.T)
    x    = np.zeros(n, dtype=np.int8)
    x[rng.choice(n, K, replace=False)] = 1
    v0   = float(x @ Q @ x)
    terms  = np.empty((0, 3), dtype=np.int32)
    coeffs = np.empty(0, dtype=np.float64)
    _, v1  = kswap_polish(x, Q, terms, coeffs)
    assert v1 <= v0 + 1e-9
