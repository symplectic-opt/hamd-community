# HAMD Research Release

Hyper-Adaptive Momentum Dynamics (HAMD)

Reduced-scope research implementation for native higher-order binary optimization.

**Noncommercial research and evaluation use only.** See `LICENSE`.

[![License: Noncommercial Research/Evaluation](https://img.shields.io/badge/License-Noncommercial-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Optional_GPU-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)](https://arxiv.org/)

---

## Overview

HAMD is a physics-inspired optimization framework for constrained binary optimization, with particular emphasis on selected higher-order binary objectives.

This repository provides a reduced-scope research implementation intended for:

- methodological inspection
- reproducibility of selected examples
- small-scale experimentation
- educational and evaluation use

The code in this release focuses on native optimization in the original variable space, followed by exact-cardinality projection and deterministic local refinement.

---

## Why native higher-order optimization?

Many practical optimization pipelines are strongest on pairwise quadratic objectives. When a problem contains cubic or other higher-order interactions, a common approach is to transform the problem into an augmented quadratic surrogate using auxiliary variables and penalty terms.

This repository is intended to support research into an alternative approach: optimizing selected higher-order objectives directly in the original variable space, then projecting to feasible binary decisions and applying local refinement.

---

## What is included in this release

This research release includes a reference implementation of core HAMD components such as:

- continuous native-space dynamics on the `[0, 1]^n` relaxation
- exact cubic gradients in the original variable space
- exact-`K` projection for hard cardinality-constrained problems
- local refinement via `K`-swap search
- small benchmark drivers for cubic HUBO and toy cubic portfolio-style problems
- reference baseline scripts for quadratized surrogate comparisons
- runnable examples and basic tests

The goal is to make the method understandable, inspectable, and runnable on public research examples.

---

## Scope of this repository

This repository is a **research release**, not a complete benchmark or deployment package.

It is intended to support:

- reproducibility of selected experiments
- method evaluation on public and small-scale instances
- inspection of the main algorithmic components
- extension by researchers interested in higher-order binary optimization

Some large-scale automation, benchmarking infrastructure, and supporting tooling used in internal experimentation are not part of this release.

---

## Benchmark context

The repository includes reduced-scope benchmark workflows and selected reference outputs associated with the manuscript.

These examples are intended to illustrate the method on public research instances and toy problem classes, rather than provide a full benchmarking framework.

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

## Installation

```bash
git clone https://github.com/symplectic-opt/hamd-community.git
cd hamd-community
pip install -e .
```

**Requirements:**

- Python >= 3.10
- PyTorch >= 2.1
- NumPy >= 1.24

---

## Quick start

```python
from hamd.core.utils import load_instance
from hamd.core.native_cubic_hamd import NativeCubicHAMD

inst = load_instance("data/cubic_hubo/cubic_n50.json")

solver = NativeCubicHAMD(
    steps=2000,
    lr=0.05,
    momentum=0.9,
    transverse=0.05,
    seed=123,
)

result = solver.solve(inst)

print("best objective:", result.best_obj)
print("selected support:", result.best_support)
```

See `examples/` and `benchmarks/` for additional usage patterns.

---

## Intended use

This repository is intended for:

- academic research
- private technical evaluation
- reproducibility exercises
- experimentation with native higher-order binary optimization methods

It is not presented as a production system, managed service, or deployment-ready package.

---

## Citation

If you use this repository in research, please cite the associated manuscript and [`CITATION.cff`](CITATION.cff).

Example:

```bibtex
@software{hamd_research_release,
  title  = {HAMD Research Release},
  author = {Author Name},
  year   = {2026},
  note   = {Reduced-scope research implementation for native higher-order binary optimization}
}
```

---

## License

This repository is released for noncommercial research and evaluation use only.

Please see:

- [`LICENSE`](LICENSE)
- [`LICENSE_SUMMARY.md`](LICENSE_SUMMARY.md)

---

## Status

This is an active research codebase. Interfaces, examples, and benchmark scripts may change as the method evolves.

Contributions, issues, and reproducibility feedback are welcome where consistent with the repository license and scope.
