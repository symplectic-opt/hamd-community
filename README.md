# HAMD Community Edition

**Hyper-Adaptive Momentum Dynamics** — open research and evaluation release  
*Noncommercial use only · See [LICENSE](LICENSE)*

---

## What is HAMD?

HAMD (Hyper-Adaptive Momentum Dynamics) is a continuous-relaxation solver for
**K-cardinality-constrained higher-order binary optimisation** problems.  It
operates natively on the original variable space — without quadratizing cubic
(degree-3) objectives — achieving structural advantages over classical SA and
Tabu Search approaches.

Core ideas:
- **Continuous adiabatic dynamics** on the [0,1]^n relaxation with leapfrog integration
- **Exact K-manifold enforcement** via top-K snap (hard cardinality, always satisfied)
- **Exact cubic gradient** computed directly on *n* variables — no auxiliary blow-up
- **Geometric curvature steering** on the K-manifold tangent plane
- **K-swap steepest-descent polish** after each restart and ILS perturbation

This Community Edition provides a clean, documented reference implementation
suitable for research, evaluation, and reproducibility of the associated preprint.

---

## What This Repository Includes

| Component | Description |
|---|---|
| `hamd/core/native_cubic_hamd.py` | `NativeCubicHAMD` — reference solver |
| `hamd/core/kswap.py` | K-swap steepest-descent polish |
| `hamd/core/projection.py` | Top-K snap, K-manifold projection |
| `hamd/core/metrics.py` | Gap %, W/T/L, summary statistics |
| `hamd/generators/cubic_hubo.py` | Sparse cubic spin-glass generator |
| `hamd/generators/cubic_portfolio_toy.py` | Cubic portfolio generator (n≤200) |
| `hamd/baselines/sa_qubo.py` | SA on Rosenberg-augmented QUBO |
| `hamd/baselines/tabu_qubo.py` | Tabu on Rosenberg-augmented QUBO |
| `benchmarks/` | Public benchmark scripts |
| `data/` | Pre-generated instances (n≤100 HUBO, n=200 portfolio) |
| `paper/paper_results_public.json` | Selected validated paper results |
| `examples/` | Worked examples |
| `tests/` | pytest test suite |

## What This Repository Does NOT Include

- Full benchmark campaign orchestration (multi-scale, multi-seed, automated reporting)
- Large-scale portfolio instances (n=500, n=1000)
- Enterprise-grade decoded-feasibility analytics
- Internal research session logs
- Production deployment, API wrappers, Docker, or telemetry hooks
- The strongest tuning presets and restart policies

For production use, customer benchmarking, or the full performance envelope,
see **HAMD Enterprise Edition** (contact below).

---

## Benchmark Summary

Results from the paper (HAMD vs SA-QUBO and Tabu-QUBO, 60 s, 3 seeds):

| Problem | HAMD | SA/Tabu | Gap |
|---|---|---|---|
| HUBO cubic n=150 | **−88 to −93** | −9 | **>10×** |
| Portfolio n=200, K=40 | **195.65** | 1,208 | **83.8 %** |
| Exact n=20/25/30 | Global opt | — | **0.00 %** |

*Community benchmarks use 30 s budget on n≤100 HUBO and n=200 portfolio.*

---

## Installation

### Option A — pip install (recommended)

```bash
pip install hamd-community
```

### Option B — from source

```bash
git clone https://github.com/symplectic-opt/hamd-community
cd hamd-community
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.1.0  
CUDA is optional — all benchmarks run on CPU.

---

## Quick Start

```python
from hamd import NativeCubicHAMD, load_instance

inst = load_instance("data/cubic_portfolio/cubicport_n200_k40.json")
solver = NativeCubicHAMD(
    n            = inst['n'],
    K            = inst['K'],
    Q_quad       = inst['Q_quad'],
    cubic_terms  = inst['cubic_terms'],
    cubic_coeffs = inst['cubic_coeffs'],
)
result = solver.solve(budget_sec=30.0, seed=42)
print(f"Best value: {result['best_value']:.4f}   K={result['k_actual']}")
```

---

## Running Benchmarks

```bash
# Smoke test (~2 min, 10s budget, 1 seed)
bash benchmarks/run_public_benchmarks.sh --quick

# Full public benchmark suite (~18 min, 30s budget, 3 seeds)
bash benchmarks/run_public_benchmarks.sh

# Individual experiments
python benchmarks/run_exact_small.py --sizes 20 25 30 --budget-sec 10
python benchmarks/run_hubo_small.py  --budget-sec 30  --seeds 3
python benchmarks/run_portfolio_small.py --budget-sec 30 --seeds 3
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Community vs Enterprise

| Feature | Community | Enterprise |
|---|---|---|
| Native cubic HAMD solver | ✓ (reference impl) | ✓ (full performance) |
| K-swap polish | ✓ | ✓ |
| ILS basin hopping | ✓ (basic) | ✓ (full tuned) |
| HUBO benchmark (n≤100) | ✓ | ✓ |
| Portfolio benchmark (n=200) | ✓ | ✓ |
| Portfolio benchmark (n=500, 1000) | — | ✓ |
| Multi-seed ablation study | — | ✓ |
| Decoded-feasibility analytics | — | ✓ |
| Full campaign orchestration | — | ✓ |
| Enterprise benchmark presets | — | ✓ |
| Deployment / API / Docker | — | ✓ |
| Commercial use permitted | — | ✓ |
| License | Noncommercial | Commercial |

---

## Citation

If you use HAMD in your research, please cite:

```bibtex
@article{hamd2026cubic,
  title   = {Hyper-Adaptive Momentum Dynamics for Native Cubic Portfolio
             Optimization: Avoiding Quadratization Distortion in
             Higher-Order Cardinality-Constrained Search},
  author  = {Symplectic Optimization},
  year    = {2026},
  note    = {Preprint}
}
```

See also [CITATION.cff](CITATION.cff).

---

## License

This software is released for **noncommercial research and evaluation use only**.  
See [LICENSE](LICENSE) for full terms.

Commercial use, production deployment, and hosted services require a separate
commercial license from Symplectic Optimization.

---

## Contact

**Commercial licensing:** contact@symplectic-opt.com  
**Issues / research questions:** GitHub Issues  
**Enterprise Edition:** contact@symplectic-opt.com
