# HAMD Community Edition

**Hyper-Adaptive Momentum Dynamics (HAMD)**  
Open research and evaluation release for native higher-order binary optimization.

**Noncommercial use only.** See [LICENSE](LICENSE).

[![License: Noncommercial Research/Evaluation](https://img.shields.io/badge/License-Noncommercial-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Optional_GPU-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)](https://arxiv.org/)

---

## What is HAMD?

HAMD is a physics-inspired optimization framework for **dense constrained binary optimization** and selected **higher-order binary objectives**.

This repository focuses on a problem class where classical QUBO-oriented heuristics often rely on **quadratization**: a higher-order binary objective is converted into an augmented quadratic surrogate with auxiliary variables and penalty terms. HAMD instead operates **natively in the original variable space**, then projects back to exact-cardinality binary decisions and applies deterministic local refinement.

The Community Edition provides a **reference implementation** suitable for:

- academic research
- technical evaluation
- reproducing selected results from the associated preprint
- experimenting with native cubic and cardinality-constrained binary optimization

This is **not** the full enterprise solver.

---

## Why native higher-order optimization matters

Many optimization pipelines are naturally strongest on **pairwise quadratic** objectives. When a problem contains cubic or other higher-order terms, a common route is to introduce auxiliary variables and reduce the problem to QUBO form.

That reduction can increase dimensionality and reshape the search landscape. In the cubic portfolio benchmark associated with the HAMD preprint, Rosenberg-style quadratization produces a **5× variable inflation**:

- original problem: `n`
- quadratized surrogate: `n + 4n = 5n`

HAMD is designed to study and exploit the alternative: **optimize the native higher-order objective directly**, then perform exact-cardinality projection and local refinement in the original space.

---

## Core ideas in this release

HAMD Community Edition includes a reference implementation of the following ingredients:

- **Continuous native-space dynamics** on the `[0, 1]^n` relaxation
- **Exact cubic gradients** in the original variable space
- **Exact-K projection** for hard cardinality-constrained problems
- **K-swap local refinement**
- **Small benchmark drivers** for cubic HUBO and cubic portfolio toy problems
- **Reference baselines** for SA and Tabu on quadratized QUBO surrogates

The goal of this repository is to make the method understandable, inspectable, and runnable on public research examples.

---

## Community Edition scope

### Included

| Component | Description |
|---|---|
| `hamd/core/native_cubic_hamd.py` | `NativeCubicHAMD` reference solver |
| `hamd/core/projection.py` | Top-K snap and K-manifold helpers |
| `hamd/core/kswap.py` | K-swap steepest-descent polish |
| `hamd/core/metrics.py` | Gap %, W/T/L, summary utilities |
| `hamd/generators/cubic_hubo.py` | Sparse cubic HUBO / spin-glass generator |
| `hamd/generators/cubic_portfolio_toy.py` | Toy cubic portfolio generator |
| `hamd/baselines/sa_qubo.py` | Simulated annealing on quadratized QUBO |
| `hamd/baselines/tabu_qubo.py` | Tabu search on quadratized QUBO |
| `benchmarks/` | Public benchmark scripts |
| `examples/` | Minimal runnable examples |
| `tests/` | Basic test suite |
| `paper/paper_results_public.json` | Selected validated reference results |

### Not included

This repository does **not** contain the full benchmark-grade or production-grade package.

Not included in Community Edition:

- full large-scale benchmark campaign orchestration
- full paper-scale automated reporting pipeline
- large enterprise benchmark instances (`n=500`, `n=1000+`)
- premium decoded-feasibility and forensic benchmark analytics
- advanced commercial tuning presets
- deployment containers, API wrappers, or cluster orchestration
- production support or commercial-use rights

For commercial use, benchmark bake-offs, or the full enterprise solver, contact the HAMD maintainer.

---

## Benchmark summary

Selected results associated with the preprint:

| Problem | HAMD | SA / Tabu on quadratized surrogate | Notes |
|---|---:|---:|---|
| Cubic HUBO pilot (`n=150`) | `-88` to `-93` | `-9` | native higher-order pilot result |
| Cubic portfolio (`n=200`, `K=40`) | `195.65` | `1208.07` median | 3-seed study |
| Exact small-size calibration (`n=20/25/30`) | exact optimum | — | `9/9` exact matches |

Community Edition includes **smaller and reduced-scope benchmark workflows** intended for research and evaluation, not the full benchmark campaign used for the manuscript.

---

## Repository layout

```text
hamd-community/
├── README.md
├── LICENSE
├── LICENSE_SUMMARY.md
├── CITATION.cff
├── requirements.txt
├── pyproject.toml
│
├── docs/
├── hamd/
│   ├── core/
│   ├── generators/
│   └── baselines/
│
├── examples/
├── benchmarks/
├── data/
├── paper/
└── tests/
```

---

## Install

```bash
git clone https://github.com/symplectic-opt/hamd-community.git
cd hamd-community
pip install -e .
```

Requires Python ≥ 3.10, PyTorch ≥ 2.1.0, NumPy ≥ 1.24.

---

## Quick start

```python
from hamd.core.utils import load_instance
from hamd.core.native_cubic_hamd import NativeCubicHAMD

inst = load_instance('data/cubic_hubo/cubic_n50.json')
solver = NativeCubicHAMD(
    n=inst['n'], K=inst['k_target'],
    cubic_terms=inst['cubic_terms'],
    cubic_coeffs=inst['cubic_coeffs'])
result = solver.solve(budget_sec=10.0, seed=42)
print(result['best_value'], result['k_actual'])
```

See `docs/quickstart.md` for a full walkthrough.

---

## Community vs Enterprise

See [`docs/community_vs_enterprise.md`](docs/community_vs_enterprise.md) for a full comparison.

---

## Citation

If you use HAMD in academic work, please cite the associated preprint. See [`CITATION.cff`](CITATION.cff).

---

## License

Noncommercial research and evaluation only. See [`LICENSE`](LICENSE) and [`LICENSE_SUMMARY.md`](LICENSE_SUMMARY.md).

---

## Contact

**Commercial licensing, enterprise evaluation, benchmark bake-offs:**  
grserb.research@gmail.com

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
