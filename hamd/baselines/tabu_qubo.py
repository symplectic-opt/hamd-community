"""
hamd.baselines.tabu_qubo
=========================
Tabu Search on a Rosenberg-augmented QUBO.

Same setting as SAOnAugmentedQUBO: operates on the augmented variable space
with the cubic objective Rosenberg-quadratized and K enforced as a soft
penalty.  Uses best-improving single-flip moves with a fixed tabu tenure.
"""

from __future__ import annotations

import time

import numpy as np


class TabuOnAugmentedQUBO:
    """
    Tabu Search on a Rosenberg-augmented QUBO.

    Parameters
    ----------
    Q_aug   : augmented QUBO matrix, shape (n_aug, n_aug)
    n_orig  : number of original (non-auxiliary) variables
    K       : target cardinality in original variable space
    seed    : random seed
    budget  : wall-clock time limit (seconds)
    tenure  : tabu tenure (auto-selected if None)
    """

    def __init__(
        self,
        Q_aug:   np.ndarray,
        n_orig:  int,
        K:       int,
        seed:    int   = 42,
        budget:  float = 30.0,
        tenure:  int   = None,
    ) -> None:
        self.Q_aug   = Q_aug
        self.n_aug   = Q_aug.shape[0]
        self.n_orig  = n_orig
        self.K       = K
        self.seed    = seed
        self.budget  = budget
        self.tenure  = tenure if tenure is not None else max(7, min(25, self.n_aug // 6))

    def solve(self) -> dict:
        """
        Run Tabu Search and return the best solution found.

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

        Q_sym    = Q + Q.T
        Qx       = Q @ x.astype(np.float64)
        val      = float(x @ Q @ x)
        best_val = val
        best_x   = x.copy()
        tabu     = np.zeros(n, dtype=np.int64)
        it       = 0
        t_start  = time.perf_counter()

        while True:
            elapsed = time.perf_counter() - t_start
            if elapsed >= self.budget:
                break
            it += 1

            best_i, best_d = -1, float('inf')
            for i in range(n):
                if tabu[i] > it:
                    continue
                d = (1 - 2 * int(x[i])) * (float(Q_sym[i] @ x) - Q[i, i])
                if d < best_d:
                    best_d = d
                    best_i = i

            if best_i < 0:
                best_i = int(np.argmin(tabu))
                best_d = (1 - 2 * int(x[best_i])) * (
                    float(Q_sym[best_i] @ x) - Q[best_i, best_i])

            x[best_i] ^= 1
            Qx         = Q @ x.astype(np.float64)
            val       += best_d
            tabu[best_i] = it + self.tenure

            if val < best_val:
                best_val = val
                best_x   = x.copy()

        runtime = time.perf_counter() - t_start
        return {
            'solver':      'Tabu-AugQUBO',
            'best_value':  best_val,
            'runtime_sec': runtime,
            'state':       best_x[:self.n_orig],
            'state_aug':   best_x.copy(),
            'k_actual':    int(best_x[:self.n_orig].sum()),
        }
