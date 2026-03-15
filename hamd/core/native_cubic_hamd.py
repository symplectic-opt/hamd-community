"""
hamd.core.native_cubic_hamd
============================
NativeCubicHAMD — community reference implementation of Hyper-Adaptive
Momentum Dynamics for K-cardinality-constrained higher-order objectives.

This is a clean, documented reference solver that demonstrates the core HAMD
ideas on cubic (degree-3) binary optimisation problems.  It is intentionally
self-contained and readable.  For best-in-class benchmark performance,
contact grserb.research@gmail.com about the Enterprise Edition.

Algorithm overview
------------------
Phase 1 — Continuous adiabatic dynamics (80 % of time budget)
  • Batch of parallel trajectories in the continuous relaxation [0,1]^n
  • Leapfrog / Störmer-Verlet integrator with adaptive damping γ
  • Exact cubic gradient + quadratic gradient (when Q_quad is provided)
  • Geometric curvature steering on the K-manifold tangent plane
  • RMSProp-style adaptive mass for per-coordinate step scaling
  • Snap-and-evaluate every 10 steps (top-K projection to binary)
  • Random restart every 200 steps with K-swap polish of the best snap

Phase 2 — ILS basin hopping (remaining 20 % of time budget)
  • Random 2-pair perturbation of the current best binary solution
  • K-swap steepest-descent polish after each perturbation
  • Random-walk acceptance (always moves; track global best separately)

Usage
-----
    from hamd import NativeCubicHAMD, load_instance

    inst = load_instance("data/cubic_portfolio/cubicport_n200_k40.json")
    solver = NativeCubicHAMD(
        n   = inst['n'],
        K   = inst['K'],
        Q_quad       = inst['Q_quad'],
        cubic_terms  = inst['cubic_terms'],
        cubic_coeffs = inst['cubic_coeffs'],
    )
    result = solver.solve(budget_sec=30.0, seed=42)
    print(result['best_value'], result['best_state'].sum())

See examples/ for worked examples on HUBO and portfolio instances.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from hamd.core.kswap import kswap_polish
from hamd.core.projection import topk_snap, k_project_grad

# Use GPU if available; falls back to CPU transparently
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NativeCubicHAMD:
    """
    Native cubic HAMD solver for K-cardinality-constrained binary optimisation.

    Supports objectives of the form

        f(x) = x^T Q_quad x  +  Σ_t  C_t x_{a_t} x_{b_t} x_{c_t}

    subject to  Σ_i x_i = K,  x_i ∈ {0, 1}.

    For pure HUBO instances (no quadratic part) set ``Q_quad=None``.

    Parameters
    ----------
    n            : number of decision variables
    K            : exact cardinality constraint (number of selected variables)
    Q_quad       : quadratic cost matrix, shape (n, n), symmetric; or None
    cubic_terms  : integer array, shape (m, 3)
    cubic_coeffs : float array, shape (m,)
    batch_size   : number of parallel trajectories (default 16)
    gamma        : initial damping coefficient (default 0.10)
    zeta         : geometric steering strength (default 0.50)
    dt           : leapfrog step size (default 0.05)
    """

    def __init__(
        self,
        n: int,
        K: int,
        cubic_terms,           # list or np.ndarray [m, 3]
        cubic_coeffs,          # list or np.ndarray [m]
        Q_quad: Optional[np.ndarray] = None,
        batch_size: int = 16,
        gamma:      float = 0.10,
        zeta:       float = 0.50,
        dt:         float = 0.05,
    ) -> None:
        self.n          = n
        self.K          = K
        self.batch_size = batch_size
        self.gamma      = gamma
        self.zeta       = zeta
        self.dt         = dt

        # Quadratic part (optional)
        if Q_quad is not None:
            Q_sym = 0.5 * (Q_quad + Q_quad.T)
            self.Q_np = Q_sym
            self.Q_t  = torch.tensor(Q_sym, dtype=torch.float64, device=DEVICE)
            self.Q_diag = np.diag(Q_sym)
            self._has_quad = True
        else:
            self.Q_np   = np.zeros((n, n), dtype=np.float64)
            self.Q_t    = torch.zeros(n, n, dtype=torch.float64, device=DEVICE)
            self.Q_diag = np.zeros(n, dtype=np.float64)
            self._has_quad = False

        # Cubic part
        cubic_terms  = np.asarray(cubic_terms,  dtype=np.int32)
        cubic_coeffs = np.asarray(cubic_coeffs, dtype=np.float64)
        self.terms_np  = cubic_terms
        self.coeffs_np = cubic_coeffs
        if len(cubic_terms) > 0:
            self.terms_t  = torch.tensor(cubic_terms,  dtype=torch.long,    device=DEVICE)
            self.coeffs_t = torch.tensor(cubic_coeffs, dtype=torch.float64, device=DEVICE)
        else:
            self.terms_t  = None
            self.coeffs_t = None

    # ─────────────────────────────────────── gradient (torch, batch) ──────

    def _grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the objective w.r.t. x (continuous), batched.

        g_i = 2 (Q_quad x)_i  +  Σ_{t ∋ i} C_t x_j x_k
        """
        # Quadratic part
        g = 2.0 * torch.mm(x, self.Q_t)  # [batch, n]

        # Cubic part: for each triple (a,b,c) and each position
        if self.terms_t is not None:
            t = self.terms_t
            c = self.coeffs_t
            batch = x.shape[0]
            for pos_a, pos_b, pos_c in [(0, 1, 2), (1, 0, 2), (2, 0, 1)]:
                prods = c * x[:, t[:, pos_b]] * x[:, t[:, pos_c]]  # [batch, m]
                idx   = t[:, pos_a].unsqueeze(0).expand(batch, -1)
                g.scatter_add_(1, idx, prods)
        return g

    # ─────────────────────────────────────── discrete evaluation (torch) ──

    def _eval_t(self, x_disc: torch.Tensor) -> torch.Tensor:
        """Evaluate objective on a batch of binary vectors."""
        vals = torch.sum(x_disc * torch.mm(x_disc, self.Q_t), dim=1)  # x^T Q x
        if self.terms_t is not None:
            a, b, c = self.terms_t[:, 0], self.terms_t[:, 1], self.terms_t[:, 2]
            cubic = self.coeffs_t * x_disc[:, a] * x_disc[:, b] * x_disc[:, c]
            vals  = vals + cubic.sum(dim=1)
        return vals

    # ─────────────────────────────────────── HVP (quadratic + penalty) ───

    def _hvp(
        self,
        v: torch.Tensor,
        x: torch.Tensor,
        pen_scale: float,
    ) -> torch.Tensor:
        """Hessian-vector product: exact quadratic + double-well penalty."""
        hvp = 2.0 * torch.mm(v, self.Q_t)
        # Double-well penalty HVP: d²/dx² [x(x-1)]² = (6x²-6x+1)·4
        hvp_pen = 4.0 * (6.0 * x ** 2 - 6.0 * x + 1.0) * v
        return hvp + pen_scale * hvp_pen

    # ─────────────────────────────────────── main solver ──────────────────

    def solve(
        self,
        budget_sec: float = 30.0,
        seed: int = 42,
    ) -> dict:
        """
        Run HAMD and return the best K-feasible binary solution found.

        Parameters
        ----------
        budget_sec : total wall-clock time budget (seconds)
        seed       : random seed for reproducibility

        Returns
        -------
        dict with keys:
            best_value  : float   best native objective found
            best_state  : ndarray binary solution, shape (n,), exactly K ones
            n_restarts  : int     number of restarts performed
            n_ils       : int     number of ILS perturbations
            runtime_sec : float   actual wall time used
            k_actual    : int     cardinality of best_state (should equal K)
        """
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        n     = self.n
        K     = self.K
        batch = self.batch_size

        # Phase split: 80% continuous dynamics, 20% ILS
        hamd_budget = 0.80 * budget_sec

        # ── Initialisation ────────────────────────────────────────────────
        x = torch.zeros(batch, n, dtype=torch.float64, device=DEVICE)
        for b in range(batch):
            idx = rng.choice(n, K, replace=False)
            x[b, idx] = 1.0
        x = x + 0.01 * torch.randn_like(x)
        x.clamp_(0.0, 1.0)

        v      = torch.zeros_like(x)
        a_vec  = torch.zeros_like(x)
        mass   = torch.ones_like(x)
        dg     = torch.full((batch, 1), self.gamma,
                            dtype=torch.float64, device=DEVICE)

        best_value: float           = float('inf')
        best_state: Optional[np.ndarray] = None
        curve: List[Tuple[float, float]] = []

        t_start    = time.perf_counter()
        step       = 0
        n_restarts = 0

        # ── Phase 1: Continuous adiabatic dynamics ────────────────────────
        while True:
            elapsed = time.perf_counter() - t_start
            if elapsed >= hamd_budget:
                break

            progress  = min(1.0, elapsed / max(hamd_budget, 1e-9))
            pen_scale = 1.0 + 4.0 * progress

            # Adaptive per-trajectory damping
            kinetic     = 0.5 * torch.mean(v ** 2, dim=1, keepdim=True)
            target_temp = 0.01 * (1.0 - progress)
            dg = torch.clamp(
                dg + self.dt * (kinetic - target_temp) / 10.0, 0.01, 0.99)

            # Leapfrog half-step in velocity
            with torch.no_grad():
                v_half = (1.0 - dg / 2.0) * v + 0.5 * self.dt * a_vec
                x      = x + self.dt * v_half
                x.clamp_(0.0, 1.0)

            # Gradient
            grad_obj = self._grad(x)
            grad_pen = 4.0 * x * (x - 1.0) * (2.0 * x - 1.0)
            grad     = grad_obj + pen_scale * grad_pen

            # HVP + geometric curvature steering
            hvp      = self._hvp(v_half, x, pen_scale)
            dot_hg   = torch.sum(hvp * grad,   dim=1, keepdim=True)
            norm_g2  = torch.sum(grad ** 2,    dim=1, keepdim=True) + 1e-9
            geom     = hvp - (dot_hg / norm_g2) * grad
            geom     = k_project_grad(geom)      # K-manifold tangent plane
            norm_geom = torch.sqrt(torch.sum(geom ** 2, dim=1, keepdim=True)) + 1e-9
            geom      = geom * torch.clamp(torch.sqrt(norm_g2) / norm_geom, max=1.0)

            # RMSProp mass + velocity update
            with torch.no_grad():
                mass  = 0.9 * mass + 0.1 * (grad ** 2)
                inv_m = 1.0 / (torch.sqrt(mass) + 1e-8)
                a_vec = (-grad + self.zeta * geom) * inv_m
                v_half = k_project_grad(v_half)  # keep velocity on tangent plane
                v = (1.0 - dg / 2.0) * v_half + 0.5 * self.dt * a_vec

            # ── Snap + evaluate every 10 steps ──────────────────────────
            if step % 10 == 0:
                x_disc = topk_snap(x, K)
                vals   = self._eval_t(x_disc)
                i_best = int(torch.argmin(vals).item())
                current = float(vals[i_best].item())
                if current < best_value:
                    best_value = current
                    best_state = x_disc[i_best].cpu().numpy().astype(np.int8)
                curve.append((elapsed, best_value))

            # ── Restart every 200 steps ──────────────────────────────────
            if step % 200 == 199:
                n_restarts += 1
                x_disc = topk_snap(x, K)
                sv     = self._eval_t(x_disc)
                ib     = int(torch.argmin(sv).item())
                if float(sv[ib].item()) < best_value:
                    best_value = float(sv[ib].item())
                    best_state = x_disc[ib].cpu().numpy().astype(np.int8)

                # K-swap polish on best found so far
                if best_state is not None:
                    polished, pol_val = kswap_polish(
                        best_state, self.Q_np, self.terms_np, self.coeffs_np)
                    if pol_val < best_value:
                        best_value = pol_val
                        best_state = polished

                # Re-initialise with fresh random K-feasible starts
                for b in range(batch):
                    idx = rng.choice(n, K, replace=False)
                    x[b] = 0.0
                    x[b, idx] = 1.0
                x     = x + 0.01 * torch.randn_like(x)
                x.clamp_(0.0, 1.0)
                v     = torch.zeros_like(x)
                mass  = torch.ones_like(x)

            step += 1

        # ── Final K-swap polish ───────────────────────────────────────────
        if best_state is not None:
            best_state, best_value = kswap_polish(
                best_state, self.Q_np, self.terms_np, self.coeffs_np)
            curve.append((time.perf_counter() - t_start, best_value))

        # ── Phase 2: ILS basin hopping ────────────────────────────────────
        n_ils   = 0
        ils_rng = np.random.RandomState(seed + 9999)
        if best_state is not None:
            ils_state = best_state.copy()
            ones_buf  = np.where(ils_state > 0.5)[0]
            zeros_buf = np.where(ils_state < 0.5)[0]

            while time.perf_counter() - t_start < budget_sec:
                out_idx = ils_rng.choice(ones_buf,  2, replace=False)
                in_idx  = ils_rng.choice(zeros_buf, 2, replace=False)
                pert    = ils_state.copy()
                pert[out_idx] = 0
                pert[in_idx]  = 1
                polished, pol_val = kswap_polish(
                    pert, self.Q_np, self.terms_np, self.coeffs_np)
                n_ils    += 1
                ils_state = polished   # random-walk acceptance
                ones_buf  = np.where(ils_state > 0.5)[0]
                zeros_buf = np.where(ils_state < 0.5)[0]
                if pol_val < best_value:
                    best_value = pol_val
                    best_state = polished.copy()
                    curve.append((time.perf_counter() - t_start, best_value))

        runtime = time.perf_counter() - t_start
        return {
            'best_value':  best_value,
            'best_state':  best_state,
            'k_actual':    int(best_state.sum()) if best_state is not None else -1,
            'n_restarts':  n_restarts,
            'n_ils':       n_ils,
            'runtime_sec': runtime,
            'curve':       curve,
        }
