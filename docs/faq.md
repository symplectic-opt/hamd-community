# Frequently Asked Questions

## Q: Why not just run SA or Tabu on the original cubic problem?

**A:** SA and Tabu are defined for quadratic (QUBO/Ising) objectives.  To apply
them to a cubic problem you must first *quadratize* it — introduce auxiliary
binary variables to reduce every cubic term to a quadratic penalty.  Rosenberg
quadratization expands a cubic instance with n binary variables and m cubic terms
to a QUBO with n+m variables and O(m) penalty terms.  For a typical problem with
m≈4n this is a 5× blowup in variable count.

The penalty constant λ must be large enough to suppress constraint violations,
which flattens the energy landscape and makes it exponentially harder to navigate.
HAMD avoids this entirely by operating natively on the original n variables.

---

## Q: What is the K-manifold and why do you project onto it?

**A:** The K-cardinality constraint means exactly K of the n binary variables must
equal 1.  The set of all binary vectors x∈{0,1}^n with ‖x‖₁=K is the
K-manifold.  During the continuous adiabatic phase HAMD operates on a relaxed
[0,1]^n domain.  At each dynamics step we project back to the K-manifold via
*top-K snap*: set the K largest components to 1 and the rest to 0.  This
guarantees feasibility at every evaluation and eliminates the need for a penalty
term.

---

## Q: What is the difference between the continuous phase and the ILS phase?

**A:** HAMD has two phases inside `solve()`:

- **Phase 1 (continuous, ~80% of budget):** Leapfrog/Störmer-Verlet integration
  of Hamiltonian dynamics on [0,1]^n with an adaptive damping coefficient
  (RMSProp-style mass tensor).  At each restart the continuous trajectory is
  snapped to {0,1}^n and polished by K-swap.

- **Phase 2 (ILS, ~20% of budget):** Iterated Local Search.  From the best
  discovered solution, repeatedly perturb (swap 2 random pairs) and re-polish
  with K-swap.  This escapes local optima that the dynamics can't overcome.

---

## Q: How do I add my own objective function?

**A:** Package your objective as a dict with these keys:

```python
objective = {
    "Q_quad": np.ndarray,          # shape (n, n) quadratic coefficients (optional)
    "cubic_terms": np.ndarray,     # shape (m, 3) index triples, int
    "cubic_coeffs": np.ndarray,    # shape (m,)  term coefficients, float
}
```

Pass it to the solver:

```python
from hamd import NativeCubicHAMD
solver = NativeCubicHAMD(n=n, K=K, **objective)
result = solver.solve(budget_sec=30)
```

The generators in `hamd.generators` produce this format and save/load via JSON.
You can also construct the dict directly for your own problem.

---

## Q: What objectives does this release support?

**A:** Any minimization objective of the form:

$$f(x) = \sum_{i,j} Q_{ij} x_i x_j + \sum_{(i,j,k)} c_{ijk} x_i x_j x_k$$

subject to $\sum_i x_i = K$, $x \in \{0,1\}^n$.

This covers: cubic HUBO, cubic portfolio optimization, 3-SAT as cubic pseudo-Boolean,
weighted MAX-k-CUT as cubic, etc.

---

## Q: Can I use HAMD on a pure quadratic (QUBO) problem?

**A:** Yes.  Set `cubic_terms = np.empty((0,3), dtype=int)` and
`cubic_coeffs = np.empty(0)`, and supply only `Q_quad`.  The solver reduces to
HAMD on a K-constrained binary quadratic, which is also a valid use case.

---

## Q: What is the scope of this release?

**A:** This repository is a reduced-scope research implementation intended for
reproducibility of selected examples, method evaluation on public and small-scale
instances, and inspection of the main algorithmic components.

Some large-scale automation, benchmarking infrastructure, and supporting tooling
used in internal experimentation are not part of this release.

---

## Q: How do I cite this work?

**A:** See [CITATION.cff](../CITATION.cff) for the machine-readable citation, or use:

```bibtex
@article{hamd_cubic_2026,
  title   = {Hyper-Adaptive Momentum Dynamics for Native Cubic Portfolio Optimization:
             Avoiding Quadratization Distortion in Higher-Order
             Cardinality-Constrained Search},
  author  = {Symplectic Optimization Lab},
  year    = {2026},
  note    = {Preprint}
}
```

---

## Q: Does HAMD require a GPU?

**A:** No.  All benchmarks in this release run on CPU.  PyTorch is required for
the tensor operations, but no CUDA is needed.  The solver will automatically use
CUDA if available (`torch.cuda.is_available()` → True) to accelerate the matrix
operations, but the results are identical.

---

## Q: What Python versions are supported?

**A:** Python ≥ 3.10.  Tested on 3.10, 3.11, 3.12.  PyTorch ≥ 2.1.0.

---

## Q: How do I report a bug or ask for help?

**A:** Open a GitHub issue in this repository with a minimal reproducible example,
the Python/PyTorch/OS versions, and the full error traceback.

For research questions or bug reports, open a GitHub issue.
