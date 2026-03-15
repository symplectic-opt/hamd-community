# Community vs Enterprise

HAMD is available in two editions:

- **HAMD Community Edition** — public, source-available, noncommercial research and evaluation release
- **HAMD Enterprise Edition** — private commercial offering for production, advanced benchmarking, and deployment

This document explains the difference.

---

## Why two editions?

HAMD Community Edition exists to:

- support the associated research paper
- let technical users inspect the method
- allow academic and noncommercial evaluation
- provide a clean reference implementation for native higher-order binary optimization

HAMD Enterprise Edition exists to:

- support commercial use
- provide the full benchmark-grade hybrid pipeline
- include advanced diagnostics, tuning, and deployment support
- preserve the highest-performance implementation for enterprise customers

The Community Edition is designed to let users **understand and evaluate HAMD**.  
The Enterprise Edition is designed to let customers **deploy and rely on HAMD**.

---

## Feature comparison

| Capability | Community Edition | Enterprise Edition |
|---|---:|---:|
| License type | Noncommercial research/evaluation | Commercial |
| Commercial use rights | No | Yes |
| Core HAMD dynamics | Yes | Yes |
| Native cubic optimization support | Yes | Yes |
| Small benchmark examples | Yes | Yes |
| Exact small-size validation | Yes | Yes |
| Public-safe cubic portfolio examples | Yes | Yes |
| Full paper-scale benchmark campaign | No | Yes |
| Large-scale benchmark instances | No | Yes |
| Advanced hybrid refinement stack | Limited | Yes |
| Full decoded-feasibility analytics | No | Yes |
| Premium tuning presets | No | Yes |
| Batch throughput tooling | Limited | Yes |
| API / service wrappers | No | Yes |
| Containerized deployment | No | Yes |
| On-prem enterprise deployment | No | Yes |
| Support / tuning / integration help | No | Yes |
| Paid benchmark bake-offs | No | Yes |

---

## What is included in Community Edition

HAMD Community Edition includes a real, runnable reference implementation for:

- native cubic HAMD solving
- exact-cardinality projection
- K-swap local refinement
- small higher-order benchmark generation
- reduced cubic portfolio workflows
- selected paper-aligned examples
- exact small-size correctness validation

It is meant for:

- academic researchers
- students
- technical evaluators
- optimization practitioners exploring the method
- prospective enterprise users who want to inspect the public reference implementation

---

## What is not included in Community Edition

The public release does **not** include the full benchmark-grade or commercial deployment stack.

Examples of features reserved for Enterprise Edition include:

- full benchmark campaign orchestration
- large benchmark instances and premium data packs
- advanced deterministic local-search stack
- premium decoded-feasibility and forensic reporting
- enterprise tuning presets
- deployment assets and service wrappers
- on-prem or private environment packaging
- commercial-use rights

---

## When Community Edition is the right fit

Community Edition is appropriate if you want to:

- read and understand the method
- reproduce selected research examples
- evaluate HAMD on toy or moderate-scale problems
- compare native higher-order optimization with quadratized baselines
- use HAMD for academic research or noncommercial experimentation

---

## When Enterprise Edition is the right fit

Enterprise Edition is appropriate if you want to:

- use HAMD in a commercial setting
- benchmark HAMD on internal business problems
- deploy HAMD in production or pre-production workflows
- run larger benchmark campaigns
- access the full hybrid optimization stack
- receive tuning, support, or integration help
- explore higher-order or dense constrained optimization at scale

---

## Commercial options

HAMD Enterprise can be offered in several forms:

- paid benchmark bake-off / proof-of-concept
- annual internal-use commercial license
- on-prem private deployment
- custom integration and formulation work

Current commercial contact:

**grserb.research@gmail.com**

---

## Frequently asked questions

### Can I use Community Edition inside my company?
Only for noncommercial research/evaluation under the license terms. Production or internal business use requires a commercial license.

### Does Community Edition include the full solver used in private benchmark campaigns?
No. It includes a real public reference implementation and a reduced benchmark subset.

### Why not release everything publicly?
The Community Edition is designed to support scientific transparency and technical evaluation while preserving a commercial path for the highest-performance implementation and deployment tooling.

### Can I start with a benchmark bake-off before licensing?
Yes. A paid benchmark bake-off / PoC is often the best starting point for enterprise evaluation.

---

## Contact

For commercial licensing, enterprise evaluation, benchmark bake-offs, or custom integration:

**grserb.research@gmail.com** Comparison

## Summary

HAMD is available in two editions:

| | Community Edition | Enterprise Edition |
|---|---|---|
| **Access** | Public (GitHub) | Private (contact us) |
| **License** | Noncommercial only | Commercial |
| **Solver** | Reference implementation | Full performance-tuned |
| **Benchmark scale** | n ≤ 100 (HUBO), n=200 (portfolio) | n up to 1000+ |
| **Campaign tooling** | Per-experiment scripts | Full automated harness |
| **Diagnostics** | Basic metrics | Decoded-feasibility analytics |
| **Ablation study** | — | Full multi-mode ablation |
| **Deployment** | — | Docker, API, serving |

---

## Community Edition: What It Proves

The Community Edition is designed to let researchers and prospect users:

1. **Reproduce** the core benchmark results from the paper at reduced scale
2. **Validate** that HAMD finds provably exact optima on small instances (n≤30)
3. **Understand** the algorithmic structure from clean, documented source code
4. **Evaluate** the conceptual advantage over SA/Tabu in a controlled setting
5. **Cite** the work with a reproducible artifact

The reference solver is genuine — uses the same continuous dynamics, K-swap
polish, and ILS as the paper — but does not include the strongest production
tuning or the full diagnostic stack.

---

## Enterprise Edition: What It Adds

The Enterprise Edition is the commercial-grade solver stack:

### Solver Performance
- Fully tuned restart policies and damping schedules
- Larger default batch sizes for GPU-optimised throughput
- Advanced ILS with deeper multi-neighbourhood oscillation
- Best-in-class K-swap implementation with Cython-accelerated inner loops

### Benchmark Coverage
- Full n=200/300/500/1000 portfolio scaling study
- n=150 HUBO flagship result reproduction
- Multi-seed statistical reporting (W/T/L, median gap, confidence intervals)
- Full ablation table (HAMD-cont, HAMD-proj, HAMD-polish, HAMD-full)

### Analytics
- Decoded-feasibility package: Rosenberg aux constraint violation analysis,
  FP/FN rates, penalty decomposition, cardinality verification
- Time-to-target (TTT) curves at 10/25/50/75/100 % of budget
- CSV/JSON output for downstream analysis

### Orchestration
- One-command full campaign reproduction: `run_all_cubic_benchmarks.sh`
- Customer bake-off harness for head-to-head comparison studies
- Sensitivity analysis scripts (λ_K, λ_Rosenberg parameter sweeps)

### Deployment
- Docker image
- REST API wrapper
- Licensing / entitlement hooks
- Customer job configuration system

---

## Use Case Guidance

| Use case | Edition |
|---|---|
| Academic research, paper writing | Community |
| Teaching / course materials | Community |
| Evaluating HAMD on your research problem | Community |
| Production deployment in a product | Enterprise |
| Customer benchmark bake-off | Enterprise |
| Internal business optimisation | Enterprise |
| Hosting as a service | Enterprise |

---

## Getting Enterprise Access

Contact: **contact@symplectic-opt.com**

Please include:
- Organisation name
- Use case description
- Deployment environment (cloud, on-prem, API service)
- Preferred benchmark instances and timeline
