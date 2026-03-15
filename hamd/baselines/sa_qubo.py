"""
hamd.baselines.sa_qubo
=======================
Simulated Annealing on a Rosenberg-augmented QUBO.

This is the canonical comparison baseline used in the HAMD paper.  SA
operates on the *augmented* variable space (n + m auxiliary variables),
where the original cubic objective has been Rosenberg-quadratized and
the K-cardinality constraint has been replaced by a soft penalty term.

This demonstrates the quadratization distortion overhead: when λ_K grows
as O(n²), the SA energy landscape is dominated (~99 %) by penalty terms
rather than objective signal, causing solver behaviour to cluster at the
same coarse local minima regardless of the underlying cubic objective.
"""

from __future__ import annotations

import time

import numpy as np


class SAOnAugmentedQUBO:
    """
    Simulated Annealing on a Rosenberg-augmented QUBO.

    Parameters
    ----------
    Q_aug   : augmented QUBO matrix, shape (n_aug, n_aug)
    n_orig  : number of original (non-auxiliary) variables
    K       : target cardinality in original variable space
    seed    : random seed
    budget  : wall-clock time limit (seconds)
    T_init  : initial SA temperature (default 5.0)
    """

    def __init__(
        self,
        Q_aug:  np.ndarray,
        n_orig: int,
        K:      int,
        seed:   int   = 42,
        budget: float = 30.0,
        T_init: float = 5.0,
    ) -> None:
        self.Q_aug  = Q_aug
        self.n_aug  = Q_aug.shape[0]
        self.n_orig = n_orig
        self.K      = K
        self.seed   = seed
        self.budget = budget
        self.T_init = T_init

    def solve(self) -> dict:
        """
        Run SA and return the best solution found.

        Returns
        -------
        dict with keys:
            best_value : float   best augmented QUBO objective
            runtime_sec: float
            state      : ndarray best solution projected to original n bits
            k_actual   : int     cardinality in original variable space
        """
        rng = np.random.RandomState(self.seed)
        n   = self.n_aug
        Q   = self.Q_aug

        x = np.zeros(n, dtype=np.int8)
        x[rng.choice(self.n_orig, self.K, replace=False)] = 1

        Qx      = Q @ x.astype(np.float64)
        val     = float(x @ Qx)
        best_x  = x.copy()
        best_val = val
        t_start = time.perf_counter()

        while True:
            elapsed = time.perf_counter() - t_start
            if elapsed >= self.budget:
                break

            progress = elapsed / self.budget
            T = max(1e-4, self.T_init * (1.0 - progress) ** 2)

            i     = int(rng.randint(0, n))
            q_row = Q[i, :] + Q[:, i]
            if x[i] == 0:
                delta = float(q_row @ x) + Q[i, i]
            else:
                delta = -(float(q_row @ x) - Q[i, i])

            if delta < 0 or rng.random() < np.exp(-delta / T):
                x[i] ^= 1
                Qx    = Q @ x.astype(np.float64)
                val  += delta
                if val < best_val:
                    best_val = val
                    best_x   = x.copy()

        runtime = time.perf_counter() - t_start
        return {
            'solver':      'SA-AugQUBO',
            'best_value':  best_val,
            'runtime_sec': runtime,
            'state':       best_x[:self.n_orig],
            'state_aug':   best_x.copy(),
            'k_actual':    int(best_x[:self.n_orig].sum()),
        }
