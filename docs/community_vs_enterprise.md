# Community vs Enterprise Comparison

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
