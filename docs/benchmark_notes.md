# Benchmark Notes

## Public Benchmark Scope

This release reproduces a subset of the paper's experimental results at reduced
scale.  Some large-scale experiments and supporting infrastructure used in
internal experimentation are not part of this release.

---

## Experiment 1 — Exact Small-Size Validation

**Purpose:** Prove correctness.  Brute-force enumerate all C(n,K) feasible
assignments and verify HAMD finds the global optimum.

**Sizes:** n=20 (K=4), n=25 (K=5), n=30 (K=6)  
**Seeds:** 3  
**Budget:** 10 s per seed

**Paper result:** 9/9 trials at 0.00 % gap (provably globally optimal).

**Expected public result:** PASS on all seeds.  Run:
```bash
python benchmarks/run_exact_small.py --sizes 20 25 30 --budget-sec 10 --seeds 3
```

---

## Experiment 2 — Small HUBO Cubic Benchmark

**Purpose:** Demonstrate HAMD's structural advantage on native cubic HUBO
over SA/Tabu operating on the Rosenberg-quadratized QUBO.

**Instances:** cubic_n50, cubic_n75, cubic_n100 (K=n//2, sparse, J∈{±1})  
**Seeds:** 3  
**Budget:** 30 s

**Paper result (n=150, 60s):** HAMD −88/−93 vs SA/Tabu −9 (>10× gap).  
**Expected public result:** HAMD significantly outperforms SA/Tabu at n=50/75/100.

**Note:** SA and Tabu solve the augmented problem (n+m variables, m≈4n), giving
them 5× the variable count but gradient signal dominated by the K-penalty.

---

## Experiment 3 — Cubic Portfolio (n=200)

**Purpose:** Demonstrate on a financially motivated cubic portfolio that HAMD
avoids the quadratization distortion trap.

**Instance:** cubicport_n200_k40.json (n=200, K=40, 800 cubic terms, n_aug=1000)  
**Seeds:** 3  
**Budget:** 30 s

**Paper result (60s, 3 seeds):** HAMD 195.65 ± 0 vs SA median 1,208, Tabu median 1,208.
Median gap: 83.8 %.  W=3, T=0, L=0.

**Expected public result (30s):** HAMD significantly outperforms SA/Tabu.
Exact numbers will differ from paper (shorter budget).

---

## Reproducing Paper Results Exactly

The paper's exact numbers (60 s budget, 3 seeds, n=150 HUBO, full ablation, n=1000
portfolio) require:
- 60 s budget
- The n=150 HUBO instance (`cubic_n150.json`) — not included in this release
- The full-scale benchmark campaign infrastructure — not part of this release

---

## Hardware Notes

All benchmarks in this release are CPU-only (no GPU required).

Paper results were produced on an AWS g5.xlarge (NVIDIA A10G + 16 vCPU).
CPU-only results will be slower and may show higher variance, but HAMD's
structural advantage over SA/Tabu is hardware-independent.
