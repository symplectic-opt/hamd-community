# Quickstart Guide

## 1. Installation

```bash
# From PyPI
pip install hamd-community

# From source
git clone https://github.com/symplectic-opt/hamd-community
cd hamd-community
pip install -e ".[bench,test]"
```

**Python:** ≥ 3.10  **PyTorch:** ≥ 2.1.0  (CUDA optional; CPU works fine)

---

## 2. Smoke Test

```bash
pytest tests/ -v
```

All 9 tests should pass in under 30 seconds.

---

## 3. First Solver Run

```python
from hamd import NativeCubicHAMD, load_instance

# Load the pre-generated n=200 cubic portfolio instance
inst = load_instance("data/cubic_portfolio/cubicport_n200_k40.json")

solver = NativeCubicHAMD(
    n            = inst['n'],           # 200 assets
    K            = inst['K'],           # select 40
    Q_quad       = inst['Q_quad'],      # Markowitz quadratic part
    cubic_terms  = inst['cubic_terms'], # sector co-movement triples
    cubic_coeffs = inst['cubic_coeffs'],
)

result = solver.solve(budget_sec=30.0, seed=42)
print(f"Objective : {result['best_value']:.4f}")
print(f"Cardinality: {result['k_actual']} (should be {inst['K']})")
print(f"Restarts  : {result['n_restarts']}")
print(f"ILS steps : {result['n_ils']}")
```

---

## 4. Run the Public Benchmark Suite

```bash
# Quick smoke test (~2 min)
bash benchmarks/run_public_benchmarks.sh --quick

# Full suite (~18 min)
bash benchmarks/run_public_benchmarks.sh
```

Logs go to `results_public/`.

---

## 5. Pure HUBO (No Quadratic Part)

```python
from hamd import NativeCubicHAMD, load_instance

inst   = load_instance("data/cubic_hubo/cubic_n100.json")
n      = inst['n']
K      = inst.get('k_target', n // 2)

solver = NativeCubicHAMD(
    n=n, K=K,
    Q_quad=None,                        # pure cubic, no quadratic
    cubic_terms=inst['cubic_terms'],
    cubic_coeffs=inst['cubic_coeffs'],
)
result = solver.solve(budget_sec=30.0, seed=42)
print(f"HUBO objective: {result['best_value']:.3f}")
```

---

## 6. Exact Ground-Truth Validation

```bash
python benchmarks/run_exact_small.py --sizes 20 25 30 --budget-sec 10 --seeds 3
```

Expected output: `✓  All exact-small tests passed.`

---

## 7. Generating New Instances

```python
from hamd.generators.cubic_hubo import generate as hubo_gen
from hamd.generators.cubic_portfolio_toy import generate as port_gen

# HUBO
hubo_gen(n_vars=50, filepath="data/cubic_hubo/cubic_n50.json")

# Portfolio (community edition: n ≤ 200)
port_gen(n=100, K=20, filepath="data/cubic_portfolio/cubicport_n100_k20.json")
```

---

## 8. Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `batch_size` | 16 | Parallel trajectories; increase for more exploration |
| `gamma` | 0.10 | Damping coefficient; higher = faster convergence, less exploration |
| `zeta` | 0.50 | Geometric steering strength; higher = more curvature guidance |
| `dt` | 0.05 | Leapfrog step size; smaller = more stable on ill-conditioned problems |
| `budget_sec` | 30.0 | Total wall-clock budget |
| `seed` | 42 | Random seed for reproducibility |
