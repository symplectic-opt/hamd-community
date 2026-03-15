# HAMD Method Overview

## Problem Class

HAMD targets K-cardinality-constrained higher-order binary optimisation:

$$\min_{x \in \{0,1\}^n} f(x) = x^\top Q x + \sum_{(a,b,c)\in T} C_{abc}\, x_a x_b x_c \quad \text{s.t.} \sum_i x_i = K$$

Applications include:
- **Portfolio selection** with cubic sector co-movement penalties (co-skewness)
- **HUBO spin glasses** — higher-order Hamiltonian ground state problems
- Any K-constrained binary problem with dense or sparse higher-order interactions

---

## Why Classical Solvers Struggle

SA and Tabu Search are QUBO solvers.  To handle cubic terms, they must first
**Rosenberg-quadratize** the objective:

Each cubic term $C_{abc}\, x_a x_b x_c$ is replaced by $C_{abc}\, w_{ab}\, x_c$ where
auxiliary variable $w_{ab}$ enforces $w_{ab} = x_a x_b$ via a quadratic penalty with
strength $\Lambda_R$.

With $m$ cubic terms, the search space grows **5×** (for $m = 4n$), and the
K-cardinality constraint penalty $\lambda_K (\sum x_i - K)^2$ must dominate in
the augmented space.  As $n$ grows, $\lambda_K \propto n^2$ — causing ~99% of the
gradient signal to be penalty noise rather than objective signal.

## How HAMD Avoids This

HAMD operates natively on the original $n$-variable space:

1. **Continuous relaxation**: $x \in [0,1]^n$, evolved by leapfrog dynamics
2. **Exact cubic gradient**: $\nabla_i f = 2(Qx)_i + \sum_{t\ni i} C_t x_j x_k$
   — computed directly, no auxiliary variables
3. **Hard K-constraint**: top-K snap projects every trajectory evaluation to an
   exactly K-feasible binary vector
4. **Geometric steering**: curvature information steers dynamics along the
   K-manifold tangent plane, biasing toward informative descent directions

## Algorithm Phases

### Phase 1 — Continuous Adiabatic Dynamics (80% of budget)

- Batch of $B$ parallel trajectories
- Leapfrog / Störmer-Verlet integrator with adaptive per-trajectory damping $\gamma_b$
- RMSProp-style adaptive mass $\mu_i \leftarrow 0.9\mu_i + 0.1 g_i^2$
- Snap + evaluate every 10 steps; restart every 200 steps with K-swap polish

### Phase 2 — ILS Basin Hopping (20% of budget)

- Random 2-pair perturbation of current best binary solution
- K-swap steepest-descent polish after each perturbation
- Random-walk acceptance (always moves; global best tracked separately)

## Complexity

| Operation | Cost |
|---|---|
| Cubic gradient (batch) | $O(B \cdot m)$ per step |
| Top-K snap | $O(B \cdot n \log n)$ per snap |
| K-swap polish | $O(K \cdot (n-K) \cdot n)$ per pass |
| ILS perturbation | $O(1)$ per perturbation |
| Total per step | $O(B \cdot (m + n))$ |

Contrast with augmented QUBO: gradient is $O(B \cdot n_{aug}^2) = O(B \cdot 25n^2)$.
