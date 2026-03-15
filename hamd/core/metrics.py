"""
hamd.core.metrics
=================
Benchmark metrics and reporting helpers.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def gap_percent(hamd_val: float, baseline_val: float) -> float:
    """
    Percentage improvement of HAMD over a baseline.

        gap = 100 × (baseline - hamd) / |baseline|

    A positive gap means HAMD found a lower (better) objective.

    Parameters
    ----------
    hamd_val     : HAMD objective value (minimisation problem)
    baseline_val : baseline objective value

    Returns
    -------
    float  percentage gap
    """
    if abs(baseline_val) < 1e-30:
        return 0.0
    return 100.0 * (baseline_val - hamd_val) / abs(baseline_val)


def wintieloss(
    hamd_vals: List[float],
    baseline_vals: List[float],
    tol: float = 1e-6,
) -> Tuple[int, int, int]:
    """
    Compute Win / Tie / Loss counts of HAMD vs a baseline across seeds.

    Parameters
    ----------
    hamd_vals     : HAMD objective values per seed
    baseline_vals : baseline objective values per seed
    tol           : absolute tolerance for declaring a tie

    Returns
    -------
    (wins, ties, losses)
    """
    wins = ties = losses = 0
    for h, b in zip(hamd_vals, baseline_vals):
        scale = max(abs(h), abs(b), 1.0)
        if h < b - tol * scale:
            wins += 1
        elif abs(h - b) <= tol * scale:
            ties += 1
        else:
            losses += 1
    return wins, ties, losses


def summary_stats(values: List[float]) -> dict:
    """
    Compute basic statistics over a list of values.

    Returns
    -------
    dict with keys: median, mean, std, min, max
    """
    arr = np.array(values, dtype=np.float64)
    return {
        'median': float(np.median(arr)),
        'mean':   float(np.mean(arr)),
        'std':    float(np.std(arr)),
        'min':    float(np.min(arr)),
        'max':    float(np.max(arr)),
    }


def print_benchmark_table(
    solvers: List[str],
    results: dict,           # {solver_name: [val_seed0, val_seed1, ...]}
    random_reference: float = None,
) -> None:
    """
    Print a formatted benchmark results table.

    Parameters
    ----------
    solvers           : ordered list of solver names
    results           : dict mapping solver name -> list of objective values per seed
    random_reference  : optional random-baseline reference value to display
    """
    header = f"{'Solver':<22}  {'Median':>12}  {'Mean':>12}  {'Std':>10}  {'[Min, Max]':>22}"
    print(header)
    print("─" * len(header))

    hamd_medians = None
    for name in solvers:
        vals = results.get(name, [])
        if not vals:
            continue
        s = summary_stats(vals)
        print(f"  {name:<20}  {s['median']:>12.4f}  {s['mean']:>12.4f}  "
              f"{s['std']:>10.4f}  [{s['min']:.4f}, {s['max']:.4f}]")
        if name.startswith('HAMD'):
            hamd_medians = np.array(vals)

    if random_reference is not None:
        print(f"\n  Random reference:  {random_reference:.4f}")

    if hamd_medians is not None:
        print()
        for name in solvers:
            if name.startswith('HAMD'):
                continue
            bvals = np.array(results.get(name, []))
            if len(bvals) != len(hamd_medians):
                continue
            w, t, l = wintieloss(hamd_medians.tolist(), bvals.tolist())
            med_gap = float(np.median([gap_percent(h, b)
                                       for h, b in zip(hamd_medians, bvals)]))
            print(f"  HAMD vs {name:<14}  W={w}  T={t}  L={l}  "
                  f"median_gap={med_gap:+.1f}%")
